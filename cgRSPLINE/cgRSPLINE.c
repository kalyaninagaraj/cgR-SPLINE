//
//  cgRSPLINE.c
//  cgRSPLINE_xcode
//
//  Created by Kalyani Nagaraj on 4/24/18.
//  Copyright © 2018 Kalyani Nagaraj. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_cdf.h>

#define norm 2.328306549295728e-10
#define m1   4294967087.0
#define m2   4294944443.0
#define a12     1403580.0
#define a13n     810728.0
#define a21      527612.0
#define a23n    1370589.0

#define seedMAX 10
#define xMAX 10
#define hgMAX 5
#define stackMAX 100

FILE *logFile, *allSols, *bestSols, *currSols, *incSols;
int probID;
int stackflag=1;
int stacktop=-1;
int stack[stackMAX][xMAX];
float probParam[20], solvParam[20];
char logPath[50]="", expName[50]="";

typedef struct {
    int seed;
    float u;
} type_RNG; /* u16807d */

typedef struct {
    int flag;
    float hgmean[hgMAX];
    float hgcov[hgMAX][hgMAX];
} type_oracle; /* oracle */

typedef struct{
    int x[xMAX];
    float hgmean[hgMAX];
    float hgcov[hgMAX][hgMAX];
    float delta[hgMAX];
    float phat;
    int m;
    int work;
    int feas; // 0: s-p feasible, 1: s-p infeasible
    int lastseed[seedMAX];
    int numCalls;
} type_cgrspline; /* cgRSPLINE */

typedef struct{
    int x[xMAX];
    float hgmean[hgMAX];
    float hgcov[hgMAX][hgMAX];
    float gbar;
    float gamma[xMAX];
    int numPoints;
    int numCalls;
} type_pli; /* PLI */


// u16807d: ================================================
void u16807d(int *seed, float *u){
    
    *u=0.0;
    
    while(*u <= 0 || *u >= 1){
        *seed = ((*seed) * 16807L) % 2147483647;
        *u = (*seed)/2147483648.0;
    }
    
    return;
}

// mrg32k3a: ===============================================
void MRG32k3a (int mseed[seedMAX], double *u)
{
    long k;
    double p1, p2;
    
    /* Component 1 */
    p1 = a12 * mseed[2] - a13n * mseed[1];
    k = p1 / m1;
    p1 -= k * m1;
    if (p1 < 0.0)
        p1 += m1;
    mseed[1] = mseed[2];
    mseed[2] = mseed[3];
    mseed[3] = p1;
    
    /* Component 2 */
    p2 = a21 * mseed[6] - a23n * mseed[4];
    k = p2 / m2;
    p2 -= k * m2;
    if (p2 < 0.0)
        p2 += m2;
    mseed[4] = mseed[5];
    mseed[5] = mseed[6];
    mseed[6] = p2;
    
    /* Combination */
    if (p1 <= p2)
        *u = (p1 - p2 + m1) * norm;
    else
        *u = (p1 - p2) * norm;
    return;
}


// genPoiss: ===============================================
// Generate Poisson random variates using cdf inverse method
int poissgen(float u, int lambda){
    float s,p;
    int x=0;
    
    p=exp(-lambda);
    s=p;
    while(u>s){
        x=x+1;
        p=p*lambda/x;
        s=s+p;
    }
    return x;
}


// sSINV: ===========================================
// Oracle for the (s,S) inventory problem
type_oracle sSINV(int x[], int m, int seed[]){
    /*   INPUT:
     param =
     param(0) = problem ID
     param(1) = problem dimension = 2
     param(2) = nseeds = 1
     param(3) = nSecMeas = 1
     param(4) = demand intensity (lambda)
     param(5) = A = fixed order cost = 32
     param(6) = C = unit order cost = 3
     param(7) = I = inventory holding cost = 1
     param(8) = PI = unit backorder cost = 5
     param(9) = warmup time = 100
     param(10)= simlength = 30
     
     x     = (s,S)
     m     = sample size
     seed  = problem seed vector
     
     OUTPUT:
     flag = 0 implies that the model parameters are feasible
          = 1 implies that the model parameters are infeasible
     hgmean = ybar (defined only if flag = 0)
     hgcov  = var(ybar) (defined only if flag = 0)
     */
    
    int i,j,k,dim;
    float lambda, A, c, h, p;
    float hg[50],hgsum[50],hg2[50][50],hgsum2[50][50];
    float u=0.0, constRHS=0.009980966674587;
    int x1,x2;
    int Ik, Jk, Dk;
    int warmuptime, simlength, yesshortage;
    float cost;
    type_oracle sSINV;
    
    
    lambda     = probParam[4];
    A          = probParam[5];
    c          = probParam[6];
    h          = probParam[7];
    p          = probParam[8];
    warmuptime = probParam[9];  //100;
    simlength  = probParam[10]; //30;
    
    dim=probParam[3]+1;
    
    x1=x[1];
    x2=x[2];
    
    
    /* Initialize output variables */
    sSINV.flag = 0;    // 0: feasible
    // 1: deterministic infeasible
    // 2: deterministic feas, but sample-path infeasible
    
    /* model parameters feasibility check */
    if(lambda < 0) {sSINV.flag = 1;}
    
    /* decision-variable feasibility check */
    if(x1<20 || x1>34 || x2<40 || x2>100 || x2-x1<=0){sSINV.flag = 1;}
    
    /* return if param or x infeasible */
    if (sSINV.flag == 1) {
        return sSINV;
    }
    
    /* Initialize output variables */
    for(j=1;j<=dim;j++){
        sSINV.hgmean[j]=0.0;
        for(k=1;k<=dim;k++){
            sSINV.hgcov[j][k]=0.0;
        }
    }
    
    for(i=1;i<=m;i++){/* repeat m times */
        Ik  = x2;     /* begin simulation with initial inventory level = S */
        hg[1]  = 0;
        hg[2]  = 0;
        
        for(k=1;k<=warmuptime+simlength;k++){
            /* simulate for warmuptime+simlength periods */
            
            cost=0.0;
            yesshortage=0;
            
            u16807d(&seed[1],&u);
            Dk = poissgen(u,lambda);
            
            if(Ik < x1){
                cost = cost + A + c*(x2-Ik);
                /* fixed order cost + unit order cost * order size*/
                Jk   = x2;
            }
            else{Jk=Ik;} /* do not order, update invetory level after review*/
            if(Jk>=Dk){  /* if inventory is available to satify all demand, */
                cost = cost + h*(Jk-Dk);
                /* add on inventory cost = h * items carried */
            }
            else{
                cost = cost + p*(Dk-Jk);
                /* add on shortage cost = p*shortage size*/
                yesshortage=1;
                /* shortage occurs if there is isn't sufficient 
                 inventory to satisfy all demand*/
            }
            
            Ik=Jk-Dk; /* Update Ik after realizing demand. 
                       Ik can be negative since orders can get backlogged*/
            
            if (k>warmuptime){
                hg[1]  += cost;
                hg[2]  += yesshortage;
            }
        }
        hg[1]=hg[1]/simlength;
        hg[2]=hg[2]/simlength;
        
        hgsum[1]=hgsum[1]+hg[1];
        hgsum[2]=hgsum[2]+hg[2];
        
        for(j=1;j<=dim;j++){
            for(k=1;k<=dim;k++){
                hg2[j][k]=hg[j]*hg[k];
            }
        }
        
        for(j=1;j<=dim;j++){
            for(k=1;k<=dim;k++){
                hgsum2[j][k]=hgsum2[j][k]+hg2[j][k];
            }
        }
    }
    
    sSINV.hgmean[0] = 1; // Number of constraints
    sSINV.hgmean[1] = hgsum[1]/m;
    sSINV.hgmean[2] = hgsum[2]/m;
    for(j=1;j<=dim;j++){
        for(k=1;k<=dim;k++){
            sSINV.hgcov[j][k]=(hgsum2[j][k]/m + sSINV.hgmean[j]*sSINV.hgmean[k])/m;
        }
    }
    sSINV.hgmean[2] -= constRHS;
    
    return sSINV;
}

// TSF: ===============================================
type_oracle TSF(int x[], int m, int seed[])
{
    /* Input:
     param = [2 4 3 1  50 1000 20 20];
     param(0) = problem ID, not used
     param(1) = problem dimension = 4
     param(2) = nseeds = 3
     param(3) = nSecMeas = 1
     param(4) = warmup time = 50
     param(5) = simulation end time = 1000
     param(6) = total service rate = 20
     param(7) = total buffer space available = 20
     
     x
     x(0) = dimension d = 4
     x(1) = service rate of first server
     x(2) = service rate of second server
     x(3) = service rate of third server
     x(3) = buffer size of second server
     
     m     = sample size
     
     seed =
     seed(0) = number of random-number seeds, nseeds = 3
     seed(1) = seed for server 1
     ...
     seed(3) = seed for server 3
     
     OUTPUT:
     flag = 0 implies that the model parameters are feasible
          = 1 implies that the model parameters are infeasible
     hgmean = ybar (defined only if flag = 0 )
     hgcov  = var(ybar) (defined only if flag = 0)
   */

    int i, j, k, dim, icount=0;
    int rate1, rate2, rate3, buf2, buf3;
    int nbuf2, nbuf3, nevent;
    float warmuptime, totaltime, ratetotal, buffertotal;
    float sum, sum2;
    float timebig, thruput, clock, tend1, tend2, tend3, timemin;
    float hgsum[hgMAX],hgsum2[hgMAX][hgMAX];
    float constRHS=5.7761;
    float u = 0.0;
    type_oracle TSF;

    dim         = probParam[3] + 1;
    warmuptime  = probParam[4];
    totaltime   = probParam[5];
    ratetotal   = floor(probParam[6]+0.5);
    buffertotal = floor(probParam[7]+0.5);
    
    /* Initialize output variables */
    TSF.flag = 0;    // 0: feasible
    // 1: deterministic infeasible
    // 2: deterministic feas, but sample-path infeasible
    
    /* model parameters feasibility check */
    if(warmuptime < 0 || totaltime <= warmuptime || ratetotal < 3  || buffertotal < 0) {TSF.flag=1;}

    /* decision-variable feasibility check */
    if(x[1] < 1 || x[2] < 1 || x[3] < 1 || x[4] < 1 || x[1] > ratetotal || x[2] > ratetotal || x[3] > ratetotal || x[4] > buffertotal - 1) {TSF.flag = 1;}
    
    /* Initialize output */
    TSF.hgmean[0] = 1; //number of constraints

    /* return if param or x infeasible */
    if (TSF.flag == 1) { return TSF; }
    
    /* Initialize loop variables */
    rate1 = x[1];
    rate2 = x[2];
    rate3 = x[3];
    buf2  = x[4];
    buf3  = buffertotal - buf2;
    
    hgsum[1] = 0.0;
    hgsum[2] = 0.0;
    for(j=1;j<=dim;j++){
        TSF.hgmean[j]=0.0;
        for(k=1;k<=dim;k++){
            hgsum2[j][k]=0.0;
            TSF.hgcov[j][k]=0.0;
        }
    }

    sum = 0;
    sum2 = 0;
    for(i=1;i<=m;i++){	// m days of simulation of the system
        icount = 0;
        
        //printf("iref = %d", i);
        
        //simulate: return throughput / (total time - warmup time)
        //initialize clock, state, and event calendar
        
        timebig = totaltime + 1;
        thruput = 0;
        clock = 0;
        nbuf2 = 0;
        nbuf3 = 0;
        
        u16807d(&seed[1], &u);
        tend1 = -logl(1.0 - u)/rate1;
        tend2 = timebig;
        tend3 = timebig;
        
        //printf("seed1, seed2, seed3 =%d, %d, %d\n", iseed[1], iseed[2], iseed[3]);
        //printf("tend1, tend2, tend3 = %Lf, %Lf, %Lf\n", tend1, tend2, tend3);
        
        while(1){
            
            //get next-event time and number
            timemin = totaltime;
            nevent = 4;
            if(tend1 <= timemin){
                timemin = tend1;
                nevent =1;
            }
            if(tend2 <= timemin){
                timemin = tend2;
                nevent =2;
            }
            if(tend3 <= timemin){
                timemin = tend3;
                nevent =3;
            }
            clock = timemin;
            
            //execute the next event
            if(nevent == 1){
                //server 1: end of service
                if(nbuf2 == buf2){ tend1 = timebig;}
                else{
                    nbuf2 = nbuf2 + 1;
                    u16807d(&seed[1], &u);
                    tend1 = clock -logl(1.0 - u)/rate1;
                    if(nbuf2 == 1){
                        u16807d(&seed[2], &u);
                        tend2 = clock -logl(1.0 - u)/rate2;
                    }
                }
            }
            else if(nevent == 2){
                //server 2: end of service
                if(nbuf3 == buf3) {tend2 = timebig;}
                else{
                    nbuf2 = nbuf2 - 1;
                    nbuf3 = nbuf3 + 1;
                    if(tend1 == timebig){
                        nbuf2 = nbuf2 + 1;
                        u16807d(&seed[1], &u);
                        tend1 = clock -logl(1.0 - u)/rate1;
                    }
                    if(nbuf2 > 0){
                        u16807d(&seed[2], &u);
                        tend2 = clock -logl(1.0 - u)/rate2;
                    }
                    else{ tend2 = timebig;}
                    if(nbuf3 == 1){
                        u16807d(&seed[3], &u);
                        tend3 = clock -logl(1.0 - u)/rate3;
                    }
                }
            }
            else if(nevent == 3){
                //server 3: end of service
                if(clock >= warmuptime) {thruput = thruput + 1;}
                nbuf3 = nbuf3 - 1;
                if(nbuf2 > 0 && tend2 == timebig){
                    nbuf2 = nbuf2 - 1;
                    nbuf3 = nbuf3 + 1;
                    if(nbuf2 > 0){
                        u16807d(&seed[2], &u);
                        tend2 = clock -logl(1.0 - u)/rate2;
                    }
                }
                if(nbuf2 < buf2 && tend1 == timebig){
                    nbuf2 = nbuf2 + 1;
                    u16807d(&seed[1], &u);
                    tend1 = clock -logl(1.0 - u)/rate1;
                    if(nbuf2 == 1){
                        u16807d(&seed[2], &u);
                        tend2 = clock -logl(1.0 - u)/rate2;
                    }
                }
                tend3 = timebig;
                if(nbuf3 > 0){
                    u16807d(&seed[3], &u);
                    tend3 = clock -logl(1.0 - u)/rate3;
                }
            }			
            else if(nevent == 4){
                // event: end of simulation
                // minimize negative throughput
                sum = sum + (-thruput / (totaltime - warmuptime));
                sum2 = sum2 + pow(-thruput / (totaltime - warmuptime),2);
                break;
            }
            icount++;
            
            //printf("\neven %d finished\n", icount);
            //printf("next event = %d, clock = %.15Lf\n", nevent, clock);
            //printf("nbuf2, nbuf3 = %d, %d\n", nbuf2, nbuf3);
            //printf("tend1, tend2, tend3 = %0.15Lf, %.15Lf, %.15Lf\n", tend1, tend2, tend3);
            //printf("seeds = %d, %d, %d\n", u1.iseed, u2.iseed, u3.iseed);
            //printf("throughput = %.15Lf\n\n", thruput);
        }
    }

    TSF.hgmean[1]   = x[1]+x[2]+x[3];
    TSF.hgmean[2]   = sum/m;
    TSF.hgcov[2][2] = (sum2/m - pow(TSF.hgmean[2],2))/m;
    TSF.hgmean[2]   = TSF.hgmean[2] + constRHS;
    
    //printf("orcflow: ghat = %Lf, ix = [%d %d %d]\n\n", orc_x.ghat, x[1], x[2], x[3]);
    
    return TSF;
}


// callOracle: ===============================================
type_oracle callOracle(int x[], int m, int seed[]){
    type_oracle orc;
    int i, probID=floor(probParam[0]+0.5), numConst=floor(probParam[3]+0.5);
    float delta, epsilon;
    
    switch(probID) {
            
        case 1 :
            orc = sSINV(x, m, seed);
            break;
        
        case 2 :
            orc = TSF(x, m, seed);
            break;

        default :
            fprintf(stderr,"Invalid problem ID!\n");
            
    }
    
    if(orc.flag==1){
        fprintf(logFile,"Invalid parameters!\n");
        return orc;
    }
    
    for(i=2;i<=numConst+1;i++){
        epsilon=fminl(fmaxl(sqrt(orc.hgcov[i][i]*m)/pow(m,0.45), -orc.hgmean[i]), sqrt(orc.hgcov[i][i]*m)/pow(m,0.1));
        delta=(log(sqrt(orc.hgcov[i][i]*m))-log(epsilon))/log(m);
        if(orc.hgmean[i] > sqrt(m*orc.hgcov[i][i])/pow(m,delta)){
            orc.flag=2;
            break;
        }
    }
    return orc;
}


// push: ===============================================
void push(int x[])
{
    int i, ctr, flag;
    
    if(stacktop==stackMAX-1)
    {
        fprintf(logFile,"\nStack is full!!");
    }
    else if(stacktop==-1){
        stacktop++;
        for(i=0;i<=x[0]+1;i++)
            stack[stacktop][i]=x[i];
        return;
    }
    else{
        for(ctr=stacktop;ctr>=0;ctr--){
            flag=0;
            for(i=0;i<=x[0]+1;i++){
                if(stack[ctr][i]!=x[i]){
                    flag=1;
                    break;
                }
            }
            if(flag==0){return;}
        }
        
        stacktop++;
        for(i=0;i<=x[0]+1;i++){
            stack[stacktop][i]=x[i];
        }
    }
}

// disp: ===============================================
void disp()
{
    int i, ctr, id;
    
    id = floor(probParam[1]+0.5);
    
    if(stacktop==-1)
    {
        fprintf(logFile, "\nStack is empty!!");
    }
    else
    {
        fprintf(logFile, "\n\nStack size = %d \n", stacktop);
        for(ctr=stacktop;ctr>=0;--ctr){
            for(i=0;i<=id;i++){
                fprintf(logFile,"%d ",stack[ctr][i]);
            }
            fprintf(logFile,"\n");
        }
    }
}

// dsort: ===============================================
void dsort(float z[], int p[]) {
    
    float tempz;
    int tempi;
    int i, j, dim;
    
    dim=p[0];
    for(i=0;i<=dim+1;i++){
        p[i]=i;
    }
    
    for(i=0;i<=dim;i++){
        for(j=i+1;j<=dim+1;j++){
            if(z[i]<z[j]){
                tempz=z[i];
                tempi=p[i];
                z[i]=z[j];
                p[i]=p[j];
                z[j]=tempz;
                p[j]=tempi;
            }
        }
    }
    
    return;
}

// getInitLocn: ===============================================
void getInitLocn(int solvSeed[], int x0[], int xmin[], int xmax[]){
    int i;
    float u;
    
    for(i=1;i<=x0[0];i++){
        u16807d(&solvSeed[i],&u);
        x0[i] = floor(xmin[i] + u*(xmax[i] - xmin[i]) + 0.5);
    }
    return;
}



// PERTURB: ===============================================
void PERTURB(int x[], float x1[], int seed){
    int x0[xMAX], id, i;
    float u=0.0;
    
    id=x[0];
    x1[0]=id;
    
    
    // PRINT TO LOG
    fprintf(logFile,"\t\t\t\tPERTURB begins ==== \n");
    // END PRINT
    
    for(i=0;i<=id;i++){
        x0[i]=x[i];
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\t\ti/p seed = %d, ", seed);
        // END PRINT
        
        u16807d(&seed, &u);
        x1[i] = x0[i] + .3*(u - 0.5);
        
        // PRINT TO LOG
        fprintf(logFile,"u = %0.15f, i = %d, x1.x[i] = % 8.4f\n", u, i, x1[i]);
        // END PRINT
        
    }
    // PRINT TO LOG
    fprintf(logFile,"\t\t\t\t==== PERTURB ends\n");
    // END PRINT
    
    return;
}



// PLI: ===============================================
type_pli PLI(float x[], int mk, int seed[]){
    
    int x0[xMAX], p[xMAX+2];
    int i, j, k, id, numConst, numSeeds, seedk[10];
    float z[xMAX+2], w[xMAX+1], wsum, ghatprev, strange, ftol;
    type_pli bestPLI;
    type_oracle x0Oracle;
//    intVect  pvect;
    
    // Initialize
    id = floor(x[0]+ 0.5);
    numConst = floor(probParam[3]+0.5);
    numSeeds = seed[0];
    ftol=solvParam[14];
    bestPLI.numPoints=0;
    bestPLI.numCalls=0;
    strange=3.145962987654;
    
    // Copy seed to seedk
    for(i=0;i<=numSeeds;i++){
        seedk[i]=seed[i];
    }
    
    // x0 = floor(x)
    x0[0] = id;
    for(i=1;i<=id;i++){
        x0[i] = floor(x[i]);
        z[i]  = x[i] - x0[i];
    }
    
    // Copy x0 to bestPLI.x
    for(i=0;i<=id;i++){
        bestPLI.x[i]=x0[i];
    }

    // Initialize bestPLI.gamma to zero
    bestPLI.gamma[0]=id;
    for(i=0;i<=id;i++){
        bestPLI.gamma[i]=0;
    }
    
    
    // Generate weights
    z[0]    = 1;
    z[id+1] = 0;
    p[0]    = id;
//    p[0]    = 1;
//    p[id+1] = id+1;
//    pvect=dsort(z, 0, id+1);
//    for(i=0;i<=id+1;i++){
//        p[i]=pvect.x[i];
//    }
    dsort(z, p);
    for(i=0;i<=id;i++){
        w[i]=z[p[i]]-z[p[i+1]];
    }
    wsum=0;
    bestPLI.gbar = 0;

    
    // PRINT TO LOG
    fprintf(logFile,"\t\t\t\t\tp z w\n");
    for(i=0;i<=id+1;i++){
        fprintf(logFile,"\t\t\t\t\t%d % 8.4f % 8.4f\n", p[i], z[i], w[i]);
    }
    fprintf(logFile,"\t\t\t\t\ti = 0, x0 = [");
    for(j=1;j<=id;j++){
        fprintf(logFile,"%d ", x0[j]);
    }
    fprintf(logFile, "], Input seed = [");
    for(j=1;j<=numSeeds;j++){
        fprintf(logFile,"%d ", seed[j]);
    }
    fprintf(logFile,"], mk = %d, ", mk);
    // END PRINT

    // Call oracle at x0
    x0Oracle = callOracle(x0, mk, seed);

    fprintf(logFile,"flag = %d\n", x0Oracle.flag);
    
    if(x0Oracle.flag != 1){bestPLI.numCalls = bestPLI.numCalls + mk;}
    if(x0Oracle.flag == 0){
        bestPLI.numPoints++;
        wsum = wsum + w[0];
        bestPLI.gbar = bestPLI.gbar + w[0]*x0Oracle.hgmean[1];
        
        // Copy x0Oracle.hgmean to bestPLI.hgmean and ghatprev
        ghatprev=x0Oracle.hgmean[1];

        for(i=0;i<=numConst+1;i++){
            bestPLI.hgmean[i] = x0Oracle.hgmean[i];
        }
        
        // Copy x0Oracle.hgcov to bestPLI.hgcov
        for(i=1;i<=numConst+1;i++){
            for(j=1;j<=numConst+1;j++){
                bestPLI.hgcov[i][j] = x0Oracle.hgcov[i][j];
            }
        }
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\t\tx0.ghat = % 8.4f, w = % 8.4f\n\t\t\t\t\tghatold = % 8.4f, output seed = [", x0Oracle.hgmean[1], w[0], ghatprev);
        for(i=1;i<=numSeeds;i++){
            fprintf(logFile,"%d ", seed[i]);
        }
        fprintf(logFile,"]\n\n");
        // END PRINT
    }
    else{
        ghatprev = 0;
        for(i=1;i<=numConst+1;i++){
            bestPLI.hgmean[i] = strange;
            // not found a feasible bestPLI.x yet.
        }
    }
    
    // Call oracle at the other id points that form the simplex
    for(i=1;i<=id;i++){
        x0[p[i]]++;
        
        // Copy seedk to seed
        for(j=1;j<=numSeeds;j++){
            seed[j]=seedk[j];
        }
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\t\ti = %d, x0 = [", i);
        for(j=1;j<=id;j++){
            fprintf(logFile,"%d ", x0[j]);
        }
        fprintf(logFile, "], Input seed = [");
        for(j=1;j<=numSeeds;j++){
            fprintf(logFile,"%d ", seed[j]);
        }
        fprintf(logFile,"], mk = %d, ", mk);
          // END PRINT
        
        x0Oracle=callOracle(x0, mk, seed);
        
        fprintf(logFile,"flag = %d\n", x0Oracle.flag);
        
        if(x0Oracle.flag != 1){bestPLI.numCalls = bestPLI.numCalls + mk;}
        if(x0Oracle.flag == 0){

            bestPLI.numPoints++;
            wsum = wsum + w[i];
            bestPLI.gbar = bestPLI.gbar + w[i]*x0Oracle.hgmean[1];
            bestPLI.gamma[p[i]] = x0Oracle.hgmean[1] - ghatprev;
            
            // PRINT TO LOG
            fprintf(logFile,"\t\t\t\t\tx0.ghat = % 8.4f, w = % 8.4f\n\t\t\t\t\tghatold = % 8.4f, output seed = [", x0Oracle.hgmean[1], w[i], ghatprev);
            for(j=1;j<=numSeeds;j++){
                fprintf(logFile,"%d ", seed[j]);
            }
            fprintf(logFile,"]\n");
            // END PRINT
            
            ghatprev = x0Oracle.hgmean[1];
            
            
            if((bestPLI.hgmean[1] == strange) || (x0Oracle.hgmean[1] < bestPLI.hgmean[1] + ftol)){
                // Save x0 to bestPLI.x
                for(j=0;j<=id;j++){
                    bestPLI.x[j]=x0[j];
                }
                
                // Copy x0Oracle.hgmean to bestPLI.hgmean
                for(j=0;j<=numConst+1;j++){
                    bestPLI.hgmean[j] = x0Oracle.hgmean[j];
                }
                
                // Copy x0Oracle.hgcov to bestPLI.hgcov
                for(j=1;j<=numConst+1;j++){
                    for(k=1;k<=numConst+1;k++){
                        bestPLI.hgcov[j][k] = x0Oracle.hgcov[j][k];
                    }
                }
                
            }
            
            //PRINT TO LOG
            fprintf(logFile,"\t\t\t\t\txbest.ghat = % 8.4f, xbest = [", bestPLI.hgmean[1]);
            for(j=1;j<=id;j++){
                fprintf(logFile,"%d ", bestPLI.x[j]);
            }
            fprintf(logFile,"]\n\n");
            // END PRINT
            
        }
    }
    
    if(wsum > ftol){
        bestPLI.gbar = bestPLI.gbar/wsum;
    }
   
    // PRINT TO LOG
    fprintf(logFile,"\t\t\t\t\tgbar = % 8.4f, numPoints = %d, numCalls = %d\n", bestPLI.gbar, bestPLI.numPoints, bestPLI.numCalls);
    // END PRINT

    return bestPLI;
}

// SPLI: ===============================================
type_cgrspline SPLI(type_cgrspline newSPLI, int mk, int seed[]){
    
    int x0[xMAX], ix1[xMAX];
    int i, j, k, t, imax, jmax, id, numConst, numSeeds, seedk[10];
    float s, s_0, c, ftol, glength;
    float x1[xMAX], xPERT[xMAX];
    type_pli bestPLI;
    type_oracle x1Oracle;
   
    // Intialize
    numSeeds=seed[0];
    id=newSPLI.x[0];
    numConst = floor(probParam[3]+0.5);
    ftol=solvParam[14];
    newSPLI.numCalls=0;
    
    imax=100;
    jmax=5;
    
    // PRINT TO LOG
    fprintf(logFile,"\t\t\tBEGIN SPLI ===\n");
    // END PRINT
    
    // Copy seed to seedk
    for(i=1;i<=numSeeds;i++){
        seedk[i]=seed[i];
    }
    
    s_0=2.0;
    c=2.0;
    for(j=0;j<=jmax;j++){
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\t=== j = %d ===\n", j);
        fprintf(logFile,"\t\t\t\ti/p seed to PERTURB = %d\n", seed[1]);
        // END PRINT
        
        PERTURB(newSPLI.x, xPERT, seed[1]);
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\tPerturbed value of x = [");
        for(i=1;i<=id;i++){
            fprintf(logFile,"%0.15f ", xPERT[i]);
        }
        fprintf(logFile,"]\n");
        // END PRINT
        
        // Save perturbed point in x1
        for(i=0;i<=id;i++){
            x1[i]=xPERT[i];
        }
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\tPLI begins ===\n");
        // END PRINT
        
        // Save seedk to seed
        for(i=1;i<=numSeeds;i++){
            seed[i]=seedk[i];
        }
        
        // Call PLI
        bestPLI = PLI(x1, mk, seed);
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\t=== PLI ends\n");
        fprintf(logFile,"\t\t\t\tgbar = % 8.4f\n\t\t\t\tgamma = [",bestPLI.gbar);
        for(i=1;i<=id;i++){
            fprintf(logFile,"% 8.4f ", bestPLI.gamma[i]);
        }
        fprintf(logFile,"]\n\t\t\t\txbest.ghat = % 8.4f\n\t\t\t\txbest = [",bestPLI.hgmean[1]);
        for(i=1;i<=id;i++){
            fprintf(logFile,"%d ", bestPLI.x[i]);
        }
        fprintf(logFile,"]\n");
        // END PRINT
        
        newSPLI.numCalls = newSPLI.numCalls + bestPLI.numCalls;
        
        // Regardless of whether numPoints=id+1 or not, update current best
        if(bestPLI.hgmean[1] < newSPLI.hgmean[1]+ftol && bestPLI.numPoints>0){
            for(i=0;i<=id;i++){
                newSPLI.x[i]=bestPLI.x[i];
            }
            for(i=0;i<=numConst+1;i++){
                newSPLI.hgmean[i]=bestPLI.hgmean[i];
            }
            for(i=0;i<=numConst+1;i++){
                for(k=0;k<=numConst+1;k++){
                    newSPLI.hgcov[i][k]=bestPLI.hgcov[i][k];
                }
            }
            push(newSPLI.x);
            disp();
        }
        
        // Return control to SPLINE if PLI does not identify id+1 feas. points
        if(bestPLI.numPoints < id+1){
            fprintf(logFile,"\n\t\t\t\tReturn control to SPLINE because PLI could not identify id+1 feasible points\n");
            return newSPLI;
        }
        
        // Perform line search only if PLI finds id+1 feasible points
        glength=0;
        for(i=1;i<=id;i++){
            glength += pow(bestPLI.gamma[i], 2);
        }
        glength=pow(glength, 0.5);
        fprintf(logFile,"\t\t\t\tglength = % 8.4f\n", glength);
        
        if(glength <= ftol){
            return newSPLI;
        }
        
        // Copy xbest to x0
        for(i=0;i<=id;i++){
            x0[i]=newSPLI.x[i];
        }
        
        for(i=0;i<=imax;i++){
            s = s_0 * pow(c, i);
            x1[0]=id;
            for(k=1;k<=id;k++){
                ix1[k]=floor(x0[k] - bestPLI.gamma[k]*s/glength + 0.5);
            }
            
            //copy iseedk to iseed
            for(k=1;k<=numSeeds;k++){
                seed[k]=seedk[k];
            }
            
            x1Oracle = callOracle(ix1, mk, seed);
            if(x1Oracle.flag != 1){newSPLI.numCalls = newSPLI.numCalls + mk;}

            //PRINT TO LOG
            fprintf(logFile,"\t\t\t\t\t== i = %d ==\n\t\t\t\t\ts = % 8.4f, ix1 = [", i, s);
            for(k=1;k<=id;k++){
                fprintf(logFile,"%d ", ix1[k]);
            }
            fprintf(logFile,"]\n\t\t\t\t\tix1.ghat = % 8.4f, flag = %d\n\n", x1Oracle.hgmean[1], x1Oracle.flag);
            //END PRINT
            
            
            if(x1Oracle.flag!=0){
                fprintf(logFile, "\t\t\t\t\tReturn control to SPLINE since line search encountered an infeasible point.\n");
                return newSPLI;
            }
            
            if((x1Oracle.hgmean[1] >= newSPLI.hgmean[1]) && (i <= 2)) {
                fprintf(logFile, "\t\t\t\t\tReturn control to SPLINE because line search did not encounter a better point right away.\n");
                return newSPLI;
            }
            if(x1Oracle.hgmean[1] >= newSPLI.hgmean[1]){
                fprintf(logFile, "\t\t\t\t\tPerform PLI again because line search encountered a better point.\n");
                break;   //If a worse solution is found later, perform PLI again.
            }
            
            // Update xbest after every step in the line search
            newSPLI.x[0]=id;
            for(k=1;k<=id;k++){
                newSPLI.x[k]=ix1[k];
            }
            for(k=0;k<=numConst+1;k++){
                newSPLI.hgmean[k]=x1Oracle.hgmean[k];
            }
            for(k=0;k<=numConst+1;k++){
                for(t=0;t<=numConst+1;t++){
                    newSPLI.hgcov[k][t]=x1Oracle.hgcov[k][t];
                }
            }
            push(newSPLI.x);
            disp();
            
        }
    }
    return newSPLI;
}


// NE: ===============================================
type_cgrspline NE(type_cgrspline bestSPLI, int mk, int seed[]){
    
    int xold[xMAX], xquad[xMAX];
    int id, i, j, k, count, numSeeds, numConst, seedk[seedMAX];
    float ftol, xqnew, a, b, y1 = 0.0, y2=0.0, y3=0.0;
    type_cgrspline bestNE;
    type_oracle xoldOracle, xquadOracle;

    // Initialize
    id        = bestSPLI.x[0];
    numSeeds  = seed[0];
    numConst  = floor(probParam[3]+0.5);
    bestNE.numCalls=0;
    ftol      = solvParam[14];
    
    // Copy iseed to iseedk
    for(i=0;i<=numSeeds;i++){
        seedk[i]=seed[i];
    }
    
    // Copy bestSPLI to bestNE and xold
    for(i=0;i<=id;i++){
        xold[i]=bestSPLI.x[i];
        bestNE.x[i]=bestSPLI.x[i];
    }
    for(i=0;i<=numConst+1;i++){
        bestNE.hgmean[i]=bestSPLI.hgmean[i];
    }
    for(i=0;i<=numConst+1;i++){
        for(j=0;j<=numConst+1;j++){
            bestNE.hgcov[i][j]=bestSPLI.hgcov[i][j];
        }
    }
    
    y2=bestNE.hgmean[1];
    xquad[0] = id;
    
    
    // PRINT TO LOG
    fprintf(logFile,"\n\t\t\t===== NE BEGINS =====\n");
    fprintf(logFile,"\t\t\tghat at center = % 8.4f, center = [", y2);
    for(i=1;i<=id;i++){
        fprintf(logFile,"%d ", bestNE.x[i]);
    }
    fprintf(logFile,"]\n");
    // END PRINT
    
    for(i=1;i<=id;i++){
        count=1;
        xold[i]=xold[i]+1;
        
        // Copy seedk to seed
        for(j=1;j<=numSeeds;j++){
            seed[j]=seedk[j];
        }
        // Call oracle
        xoldOracle = callOracle(xold, mk, seed);
        if(xoldOracle.flag != 1){bestNE.numCalls += mk;}
        if(xoldOracle.flag == 0){
            y1=xoldOracle.hgmean[1];
            count++;
            if(xoldOracle.hgmean[1]<bestNE.hgmean[1] + ftol){
                // Copy xold to bestNE
                for(j=0;j<=id;j++){
                    bestNE.x[j]=xold[j];
                }
                for(j=0;j<=numConst+1;j++){
                    bestNE.hgmean[j]=xoldOracle.hgmean[j];
                }
                for(k=0;k<=numConst+1;k++){
                    for(j=0;j<=numConst+1;j++){
                        bestNE.hgcov[k][j]=xoldOracle.hgcov[k][j];
                    }
                }
                push(bestNE.x);
                disp();
                
            }
        }
        
        xold[i]=xold[i]-2;
        
        // Copy seedk to seed
        for(j=1;j<=numSeeds;j++){
            seed[j]=seedk[j];
        }
        
        // Call oracle
        xoldOracle = callOracle(xold, mk, seed);
        if(xoldOracle.flag != 1){bestNE.numCalls += mk;}
        if(xoldOracle.flag == 0){
            y3=xoldOracle.hgmean[1];
            count++;
            if(xoldOracle.hgmean[1]<bestNE.hgmean[1] + ftol){
                // Copy xold to bestNE
                for(j=0;j<=id;j++){
                    bestNE.x[j]=xold[j];
                }
                for(j=0;j<=numConst+1;j++){
                    bestNE.hgmean[j]=xoldOracle.hgmean[j];
                }
                for(k=0;k<=numConst+1;k++){
                    for(j=0;j<=numConst+1;j++){
                        bestNE.hgcov[k][j]=xoldOracle.hgcov[k][j];
                    }
                }
                push(bestNE.x);
                disp();
            }
            
        }
        
        xold[i]=xold[i]+1;
        
        
        xqnew=xold[i];
        
        // Quadratic search
        if(count==3){
            a = (y1+y3)/2.0 - y2;
            b = (y1-y3)/2.0;
            if ((a-0.00005) > 0){
                xqnew = xold[i] - (b / (a + a)) + 0.5;
            }
            
            // PRINT TO LOG
            fprintf(logFile,"\t\t\t\ti = %d, a = % 8.4f, b = % 8.4f, xqnew = % 8.4f\n", i, a, b, xqnew);
            fprintf(logFile,"\t\t\t\ty2 = % 8.4f, y1 = % 8.4f, y3 = % 8.4f\n", y2, y1, y3);
            // END PRINT
            
        }
        if(fabsl(xqnew) < 2147483646.0) {
            xquad[i] = xqnew;
        }
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t\txold[%d] = %d, xquad[%d] = %d\n\n", i, xold[i], i, xquad[i]);
        // END PRINT
    }
    
    // Copy iseedk to iseed
    for(i=1;i<=numSeeds;i++){
        seed[i]=seedk[i];
    }
    
    // Call oracle at xquad
    xquadOracle = callOracle(xquad, mk, seed);
    if(xquadOracle.flag != 1){bestNE.numCalls += mk;} //oracle performed mk replications
    if(xquadOracle.flag==0){
        if(xquadOracle.hgmean[1] < bestNE.hgmean[1] + ftol){
            // Copy xquad to bestNE
            for(i=0;i<=id;i++){
                bestNE.x[i]=xquad[i];
            }
            for(j=0;j<=numConst+1;j++){
                bestNE.hgmean[j]=xquadOracle.hgmean[j];
            }
            for(k=0;k<=numConst+1;k++){
                for(j=0;j<=numConst+1;j++){
                    bestNE.hgcov[k][j]=xquadOracle.hgcov[k][j];
                }
            }
        }
    }
    
    // PRINT TO LOG
    fprintf(logFile,"\t\t\t\txquad.ghat = % 8.4f, xquad = [", xquadOracle.hgmean[1]);
    for(i=1;i<=id;i++){
        fprintf(logFile,"%d ", xquad[i]);
    }
    fprintf(logFile,"]\n");
    // END PRINT
    
    return bestNE;
}



// SPLINE: ===============================================
type_cgrspline SPLINE(int x0[], int RAnum, int mk, int bk, int seed[]){
    
    int i, j, k, id, xinit[10], brkFlag=0, stackctr;
    int numCalls=0, numSeeds, numConst, seedk[10];
    float ftol, initObj[5], initCov[5][5];
    type_cgrspline bestNE, bestSPLI;
    type_oracle x0Oracle;
   
    // Initialize
    id       = x0[0];
    numSeeds = seed[0];
    numConst = floor(probParam[3]+0.5);
    ftol     = solvParam[14];
    
    // Copy iseed to iseedk
    for(i=1;i<=numSeeds;i++){
        seedk[i]=seed[i];
    }

    // Copy x0 to bestNE.x
    for(i=0;i<=id;i++){
        bestNE.x[i]=x0[i];
    }
    // PRINT TO LOG
    fprintf(logFile,"\tInput seed = [");
    for(j=1;j<=numSeeds;j++){
        fprintf(logFile,"%d ", seed[j]);
    }
    fprintf(logFile,"]\n");
    fprintf(logFile,"\n\tInitial solution x = [ ");
    for(i=1;i<=id;i++){
        fprintf(logFile,"%d ", bestNE.x[i]);
    }
    fprintf(logFile,"]\n\n");
    // END PRINT
    
    // =============================================
    
    stackctr=stacktop;
    while(1){
        // Copy iseedk to iseed
        for(i=1;i<=numSeeds;i++){
            seed[i]=seedk[i];
        }
        
        x0Oracle = callOracle(bestNE.x, mk, seed);
        if(x0Oracle.flag != 1){numCalls += mk;}
        if(x0Oracle.flag == 0){
            if(RAnum==1){
                push(bestNE.x);
                //disp();
            }
            break;
        }
        
        // Otherwise initial solution is infeasible
        fprintf(logFile, "\tInfeasible initial solution: ");
        
        if(stackctr==-1){  // Reached bottom of stack
            fprintf(logFile, "Reached bottom of stack.\n");
            stackflag=-1;
            // All visited solutions are infeasible at current sample size m

            // Copy x0Oracle to bestNE
            for(j=0;j<=numConst+1;j++){
                bestNE.hgmean[j]=x0Oracle.hgmean[j];
                for(k=0;k<=numConst+1;k++){
                    bestNE.hgcov[j][k]=x0Oracle.hgcov[j][k];
                }
            }
            bestNE.numCalls=numCalls;
            
            return bestNE; // Return control to R-SPLINE, which in turn returns
                           // control to cgRSPLINE so that the local search may
        }                  // be restarted.
        
        // Copy top of stack to bestNE
        for(i=0;i<=id+1;i++)
            bestNE.x[i]=stack[stackctr][i];
        stackctr--;
        
        
        // PRINT TO LOG
        fprintf(logFile," Next point in stack = [ ");
        for(i=1;i<=id;i++){
            fprintf(logFile,"%d ", bestNE.x[i]);
        }
        fprintf(logFile,"], Input seed = [");
        for(i=1;i<=numSeeds;i++){
            fprintf(logFile,"%d ", seedk[i]);
        }
        fprintf(logFile,"]\n");
        // END PRINT
        
    }
    
    
    // ===============================================
    
    // Copy x0Oracle to bestNE
    // Save bestNE to xinit
    for(i=0;i<=id;i++){
        xinit[i]=bestNE.x[i];
    }
    for(j=0;j<=numConst+1;j++){
        bestNE.hgmean[j]=x0Oracle.hgmean[j];
        initObj[j]=bestNE.hgmean[j];
    }
    for(k=0;k<=numConst+1;k++){
        for(j=0;j<=numConst+1;j++){
            bestNE.hgcov[k][j]=x0Oracle.hgcov[k][j];
            initCov[k][j]=bestNE.hgcov[k][j];
        }
    }
    
    // PRINT TO LOG
    fprintf(logFile,"\n\tInitial solution x = [ ");
    for(i=1;i<=id;i++){
        fprintf(logFile,"%d ", bestNE.x[i]);
    }
    fprintf(logFile,"]\n\tEstimated obj function value at x = % 8.4f\n", x0Oracle.hgmean[1]);
    fprintf(logFile,"\tNum. calls to oracle = %d\n", numCalls);
    fprintf(logFile,"\t===== BEGIN SPLINE LOOP ===\n");
    // END PRINT
    
    
    for(i=1;i<=bk;i++){
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t=== bk = %d ===\n", i);
        // END PRINT
        
        // Copy seedk to seed
        for(j=1;j<=numSeeds;j++){
            seed[j]=seedk[j];
        }
        
        // Call SPLI
        bestSPLI = SPLI(bestNE, mk, seed);
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t==== SPLI ends\n\t\t\tx = [");
        for(j=1;j<=id;j++){
            fprintf(logFile,"%d ", bestSPLI.x[j]);
        }
        fprintf(logFile,"]\n\t\t\tx.ghat = % 8.4f\n", bestSPLI.hgmean[1]);
        fprintf(logFile,"\t\t\tiseed = [");
        for(j=1;j<=numSeeds;j++){
            fprintf(logFile,"%d ", seed[j]);
        }
        fprintf(logFile,"]\n");
        // END PRINT
        
        
        // Copy seedk to seed
        for(j=1;j<=numSeeds;j++){
            seed[j]=seedk[j];
        }
        
        // Call NE
        bestNE = NE(bestSPLI, mk, seed);
        
        // PRINT TO LOG
        fprintf(logFile,"\t\t\t==== NE ends\n\t\t\tx = [");
        for(j=1;j<=id;j++){
            fprintf(logFile,"%d ", bestNE.x[j]);
        }
        fprintf(logFile,"]\n\t\t\tx.ghat = % 8.4f\n", bestNE.hgmean[1]);
        fprintf(logFile,"\t\t\tiseed = [");
        for(j=1;j<=numSeeds;j++){
            fprintf(logFile,"%d ", seed[j]);
        }
        fprintf(logFile,"]\n");
        // END PRINT
        
        // Update budget expended
        numCalls += bestSPLI.numCalls + bestNE.numCalls;
        
        // PRINT TO LOG
        fprintf(logFile,"\n\n\t\tSPLINE ncalls = prev ncalls + SPLI calls + NE calls = %d\n\n", numCalls);
        // END PRINT
        
        // Stop if bestNE.x[j]==bestSPLI.x[j]
        for(j=1;j<=id;j++){
            if(bestNE.x[j]!=bestSPLI.x[j]){
                brkFlag=1; // not equal
                break;
            }
        }
        if(brkFlag==0){
//            fprintf(logFile, "\n\t\tSPLINE ended at bk=%d since NE and SPLI returned the same solution\n\n", i);
            break;
        }
        
    }
    
    if(initObj[1] < bestNE.hgmean[1] + ftol ){
        for(i=0;i<=id;i++){
            bestNE.x[i]=x0[i];
        }
        for(j=0;j<=numConst+1;j++){
            bestNE.hgmean[j]=initObj[j];
        }
        for(k=0;k<=numConst+1;k++){
            for(j=0;j<=numConst+1;j++){
                bestNE.hgcov[k][j]=initCov[k][j];
            }
        }
    }
    
    bestNE.numCalls=numCalls;
    return bestNE;
}

// RSPLINE: ===============================================
type_cgrspline RSPLINE(int x0[], int seed[], int budg, float alpha, int r){
    
    int i, j, k, id, numSeeds, numConst, numCalls;
    int kmax, mk, bk, yesgeom;
    int xk[xMAX], lastseed[seedMAX];
    float delta0;
    float epsilon[hgMAX], delta[hgMAX], phat;
    type_cgrspline bestSPLINE, allSPLINE[stackMAX];
    
    // Problem parameters
    id       = floor(probParam[1]+0.5);
    numSeeds = floor(probParam[2]+0.5);
    numConst = floor(probParam[3]+0.5);
    
    // Solver parameters
    kmax    = floor(solvParam[2]+0.5);
    delta0  = solvParam[3];
    yesgeom = floor(solvParam[4]+0.5);
    // ftol = solvParam[17];
    
    // Intialize
    numCalls=0;
    stacktop=-1;
    
    // Save x0 to bestSPLINE
    for(i=0;i<=id;i++){
        bestSPLINE.x[i]=x0[i];
    }
    
    k=1;
    stackflag=1;
    // Retrospective Iterations
    while(numCalls<budg && k<=kmax){
        
        // PRINT TO LOG
        fprintf(logFile,"==== RETROSPECTIVE ITERATION k = %d in restart r = %d ====\n", k, r);
        // END PRINT
        
        if(yesgeom == 1) {
            // mk, bk increase geometrically
            mk = ceil(solvParam[5] * pow(solvParam[7],k));
            bk = ceil(solvParam[6] * pow(solvParam[7],k));
        }
        else {
            // mk, bk increase polynomially
            mk = solvParam[5] * ceil(pow(k,solvParam[7]));
            bk = solvParam[6] * ceil(pow(k,solvParam[7]));
        }
        
        // PRINT TO LOG
        fprintf(logFile,"mk = %d, bk = %d\n", mk, bk);
        fprintf(logFile,"===== BEGIN SPLINE =====\n");
        // END PRINT
        
        for(i=0;i<=id;i++){
            xk[i]=bestSPLINE.x[i];	// Save sol. from previous call to SPLINE
        }
        
        // Copy seed to lastseed
        fprintf(logFile,"\tPrevious seed = [");
        for(i=0;i<=numSeeds;i++){
            lastseed[i]=seed[i];
            fprintf(logFile, "%d ", lastseed[i]);
        }
        fprintf(logFile, "]\n");
        
        // Call SPLINE
        bestSPLINE = SPLINE(xk, k, mk, bk, seed);
        
        // Copy lastseed to bestSPLINE.lastseed
        fprintf(logFile,"\tPrevious seed = [");
        for(i=0;i<=numSeeds;i++){
            bestSPLINE.lastseed[i]=lastseed[i];
            fprintf(logFile, "%d ", bestSPLINE.lastseed[i]);
        }
        fprintf(logFile, "]\n");
        
        
        // Update total R-SPLINE budget expended
        numCalls += bestSPLINE.numCalls;
        
        phat=1;
        delta[0]=numConst;
        for(i=2;i<=numConst+1;i++){
            epsilon[i]=fmin(fmax(sqrt(bestSPLINE.hgcov[i][i]*mk)/pow(mk,0.45), -bestSPLINE.hgmean[i]), sqrt(bestSPLINE.hgcov[i][i]*mk)/pow(mk,0.1));
            delta[i]=(log(sqrt(bestSPLINE.hgcov[i][i]*mk))-log(epsilon[i]))/log(mk);
            if(isnan(delta[i])){delta[i]=delta0;}
            if(isinf(1.0/bestSPLINE.hgcov[i][i])){
                phat=0.0;
            }else{
                phat = phat * gsl_cdf_tdist_P ( pow(mk,(0.5-delta[i]))- bestSPLINE.hgmean[i]/sqrt(bestSPLINE.hgcov[i][i]), mk-1 );
            }

        }

        bestSPLINE.m=mk;
        bestSPLINE.phat=phat;

        if(stackflag == -1){		// No feasible solution found
            
            // Save bestSPLINE to allRSPLINE[k]
            for(i=0;i<=id;i++){
                allSPLINE[k-1].x[i]=bestSPLINE.x[i];
            }
            for(i=0;i<=numConst+1;i++){
                allSPLINE[k-1].hgmean[i]=bestSPLINE.hgmean[i];
                allSPLINE[k-1].delta[i]=delta[i];
                for(j=0;j<=id;j++){
                    allSPLINE[k-1].hgcov[i][j]=bestSPLINE.hgcov[i][j];
                }
            }
            allSPLINE[k-1].phat=phat;
            allSPLINE[k-1].m=mk;
            allSPLINE[k-1].work=numCalls;
            allSPLINE[k-1].feas=1;

            
            // PRINT TO LOG
            if(k==1){
                fprintf(logFile, "Initial solution is infeasible!!!\n");
            }else{
                fprintf(logFile, "All solutions in stack are infeasible!!!\n");
            }
            // END PRINT
            
            // WRITE allRSPLINE TO FILE
            // Write k_r first.
            // Then enter all k_r entries of allRSPLINE.
            // Write a flag that indicates end of rth restart.
            fwrite (&r, sizeof(int), 1, allSols);
            fwrite (&k, sizeof(int), 1, allSols); //    k_r
            fwrite (&budg, 1, sizeof(int), allSols); // b_r
            fwrite (&alpha, 1, sizeof(float), allSols); // alphar
            fwrite (x0, sizeof(int), id+1, allSols); // x0[i]
            fwrite (&allSPLINE, sizeof(allSPLINE[1]), k, allSols); // allRSPLINE[k]
            
            bestSPLINE.feas=1;
            bestSPLINE.work=numCalls;

            return bestSPLINE;
        }
        
        // Display all solutions in stack
        disp();
        
        // Save bestSPLINE to allRSPLINE[k-1]
        for(i=0;i<=id;i++){
            allSPLINE[k-1].x[i]=bestSPLINE.x[i];
        }
        for(i=0;i<=numConst+1;i++){
            allSPLINE[k-1].hgmean[i]=bestSPLINE.hgmean[i];
            allSPLINE[k-1].delta[i]=delta[i];
            for(j=0;j<=id;j++){
                allSPLINE[k-1].hgcov[i][j]=bestSPLINE.hgcov[i][j];
            }
        }
        allSPLINE[k-1].phat=phat;
        allSPLINE[k-1].m=mk;
        allSPLINE[k-1].work=numCalls;
        allSPLINE[k-1].feas=0;

    
        // PRINT TO LOG
        fprintf(logFile,"===== SPLINE ENDED =====\nncalls so far = %d, xbest.ghat = % 8.4f, xbest = [", bestSPLINE.numCalls, bestSPLINE.hgmean[1]);
        for(i=1;i<=id;i++){
            fprintf(logFile,"%d ", bestSPLINE.x[i]);
        }
        fprintf(logFile,"]\n\n");
        // END PRINT
    
        fprintf(logFile,"\n\n==== END RETROSPECTIVE ITERATION k = %d in restart r = %d ====\n\n", k, r);
        fprintf(logFile,"\nR-SPLINE ended because numcalls = %d >= budg = %d, or k = %d > kmax = %d\n\n", numCalls, budg, k, kmax);
//        numCalls<budg && k<=kmax
        printf("\n\n==== END RETROSPECTIVE ITERATION k = %d in restart r = %d ====\n\n", k, r);
        printf("\nR-SPLINE ended because numcalls = %d >= budg = %d, or k = %d > kmax = %d\n\n", numCalls, budg, k, kmax);
        
        k++;
    }
//    bestSPLINE.numCalls=numCalls;  //Now maintained by bestSPLINE.work
    

    // WRITE allRSPLINE TO FILE
    // Write k_r first.
    // Then enter all k_r entries of allRSPLINE.
    // Write a flag that indicates end of rth restart.
    k--;
    fwrite (&r, sizeof(int), 1, allSols);
    fwrite (&k, 1, sizeof(int), allSols); // k_r
    fwrite (&budg, 1, sizeof(int), allSols); // b_r
    fwrite (&alpha, 1, sizeof(alpha), allSols); // alphar
    fwrite (x0, sizeof(int), id+1, allSols); // x0[i]
    fwrite (&allSPLINE, sizeof(allSPLINE[1]), k, allSols); // allRSPLINE[k]
    
    bestSPLINE.feas=0;
    bestSPLINE.work=numCalls;

    return bestSPLINE;
}


// cgRSPLINE: ===============================================
void cgRSPLINE(int xmin[], int xmax[], int probSeed[], int solvSeed[], int budg){
   
    int i, j, r, id, x0[xMAX], repCount, repFlag, numSeeds, numConst, totWork=0;
    int br, yesgeom, inc_good, curr_good, tFlag=-1;
    float alphar, delta0, epsilon[xMAX], delta[xMAX], timeElapsed;
    char logFN[100]="", allSolsFN[100]="", bestSolsFN[100]="", currSolsFN[100]="", incSolsFN[100]="";
    type_cgrspline bestRSPLINE, allInc[stackMAX], allCurr[stackMAX], allBest[stackMAX];
    type_oracle xOracle;
    clock_t tic,toc;
  
    tic= clock();
    
    // Problem parameters
    id       = floor(probParam[1]+0.5);
    numSeeds = floor(probParam[2]+0.5);
    numConst = floor(probParam[3]+0.5);
    
    // Solver parameters
    delta0  = solvParam[3];
    yesgeom = floor(solvParam[9]+0.5);
    
    // Open files for printing
    strcpy(logFN, logPath);
    strcat(logFN, expName);
    strcat(logFN, ".txt");
    logFile = fopen(logFN, "w");
    if (logFile == NULL) {
        fprintf(stderr, "Can't open log file\n");
        return;
    }
    
    strcpy(allSolsFN, logPath);
    strcat(allSolsFN, "allSols");
    strcat(allSolsFN, "_");
    strcat(allSolsFN, expName);
    strcat(allSolsFN, ".dat");
    allSols = fopen (allSolsFN, "w");
    if (allSols == NULL)
    {
        fprintf(stderr, "\nError opening allSols.dat\n");
        return;
    }
    
    strcpy(bestSolsFN, logPath);
    strcat(bestSolsFN, "bestSols");
    strcat(bestSolsFN, "_");
    strcat(bestSolsFN, expName);
    strcat(bestSolsFN, ".dat");
    bestSols = fopen (bestSolsFN, "w");
    if (bestSols == NULL)
    {
        fprintf(stderr, "\nError opening bestSols.dat\n");
        return;
    }

//    strcpy(currSolsFN, logPath);
//    strcat(currSolsFN, "currSols");
//    strcat(currSolsFN, "_");
//    strcat(currSolsFN, expName);
//    strcat(currSolsFN, ".dat");
//    currSols = fopen (currSolsFN, "w");
//    if (currSols == NULL)
//    {
//        fprintf(stderr, "\nError opening bestSols.dat\n");
//        return;
//    }
//
//    
//    strcpy(incSolsFN, logPath);
//    strcat(incSolsFN, "incSols");
//    strcat(incSolsFN, "_");
//    strcat(incSolsFN, expName);
//    strcat(incSolsFN, ".dat");
//    incSols = fopen (incSolsFN, "w");
//    if (incSols == NULL)
//    {
//        fprintf(stderr, "\nError opening incSols.dat\n");
//        return;
//    }


    r=1;
    repCount=1;
    x0[0]=id;
    while(totWork<budg){
        
        fprintf(logFile,"\n\n\n\n======================== RESTART %d ===========================\n", r);
        
        // I. Generate a start solution X_r
        getInitLocn(solvSeed, x0, xmin, xmax);

        // II. Calculate br and alphar
        if(yesgeom==0){
            br=ceil(solvParam[9]*pow(r,solvParam[10]));
            // polynomial growth of restart budget
        }else{
            br=ceil(solvParam[9]*pow(solvParam[10],r));
            // geometric growth of restart budget
        }
        alphar=solvParam[11]*(1-pow(solvParam[12],1+r));
        // require Y_r to be feas with prob at least alphar
        // alphar = solvParam[11]*(1-1/log(budgr));  // Raghu's alphar
        

        // III. Run R-SPLINE
        repFlag=0;
        bestRSPLINE = RSPLINE(x0, probSeed, br, alphar, r);
        fprintf(logFile,"============ END R-SPLINE ========== \n\n");

        // IV. Update data structures
        // Update allCurr for the current solution: allCurr[r]=Y_r
        for(i=0;i<=id;i++){
            allCurr[r-1].x[i]=bestRSPLINE.x[i];
        }
        for(i=0;i<=numConst+1;i++){
            allCurr[r-1].hgmean[i]=bestRSPLINE.hgmean[i];
            for(j=0;j<=numConst+1;j++){
                allCurr[r-1].hgcov[i][j]=bestRSPLINE.hgcov[i][j];
            }
        }
        for(i=0;i<=numSeeds;i++){
            allCurr[r-1].lastseed[i]=bestRSPLINE.lastseed[i];
        }
        allCurr[r-1].m=bestRSPLINE.m;
        allCurr[r-1].phat=bestRSPLINE.phat;
        allCurr[r-1].feas=bestRSPLINE.feas;
        
        
        // PRINT TO LOG
        fprintf(logFile,"x=[");
        for(i=0;i<=id;i++){
            fprintf(logFile,"%d ", bestRSPLINE.x[i]);
        }fprintf(logFile,"]\nhgmean = [");
        for(i=0;i<=numConst+1;i++){
            fprintf(logFile,"%10.4f ", bestRSPLINE.hgmean[i]);
        }fprintf(logFile,"]\nhgvar = [");
        for(j=0;j<=numConst+1;j++){
            fprintf(logFile,"%10.4f ", bestRSPLINE.hgcov[j][j]);
        }fprintf(logFile,"]\nlast seed = [");
        for(i=0;i<=numSeeds;i++){
            fprintf(logFile,"%d ", bestRSPLINE.lastseed[i]);
        }fprintf(logFile,"]\nout seed = [");
        for(i=0;i<=numSeeds;i++){
            fprintf(logFile,"%d ", probSeed[i]);
        }fprintf(logFile,"]\nm=%d, phat=%6.4f, feas=%d\n\n",bestRSPLINE.m,bestRSPLINE.phat,bestRSPLINE.feas );
        // END PRINT
        
        // V. Update budget expended
        totWork+=bestRSPLINE.work;
        
        // VI. Compare current solution and incumbent
        if(r==1){ // First restart
            // Update allInc, allBest
            for(i=0;i<=id;i++){
                allInc[r-1].x[i]=NAN;
                allBest[r-1].x[i]=allCurr[r-1].x[i];
            }
            for(i=0;i<=numConst+1;i++){
                allInc[r-1].hgmean[i]=NAN;
                allBest[r-1].hgmean[i]=allCurr[r-1].hgmean[i];
                for(j=0;j<=numConst+1;j++){
                    allInc[r-1].hgcov[i][j]=NAN;
                    allBest[r-1].hgcov[i][j]=allCurr[r-1].hgcov[i][j];
                }
            }
            for(i=0;i<=numSeeds;i++){
                allInc[r-1].lastseed[i]=NAN;
                allBest[r-1].lastseed[i]=allCurr[r-1].lastseed[i];
            }
            allInc[r-1].m=NAN;
            allBest[r-1].m=allCurr[r-1].m;
            allInc[r-1].phat=NAN;
            allBest[r-1].phat=allCurr[r-1].phat;
            allInc[r-1].feas=NAN;
            allBest[r-1].feas=allCurr[r-1].feas;

            
            if(allCurr[r-1].feas==1){  // Current R-SPLINE call returned a
                                       // sample-path infeasible solution
                repCount++; // Repeat restart r
                repFlag=1;
            }
            
        }
        else { // r>=2
            
            // Set allInc(r) = allBest(r-1)
            for(i=0;i<=id;i++){
                allInc[r-1].x[i]=allBest[r-2].x[i];
            }
            for(i=0;i<=numConst+1;i++){
                allInc[r-1].hgmean[i]=allBest[r-2].hgmean[i];
                for(j=0;j<=numConst+1;j++){
                    allInc[r-1].hgcov[i][j]=allBest[r-2].hgcov[i][j];
                }
            }
            for(i=0;i<=numSeeds;i++){
                allInc[r-1].lastseed[i]=allBest[r-2].lastseed[i];
            }
            allInc[r-1].m=allBest[r-2].m;
            allInc[r-1].phat=allBest[r-2].phat;
            allInc[r-1].feas=allBest[r-2].feas;
            
            
            
            // POSSIBILITY I.
            // *** WILL NEVER OCCUR ***
            // Previous RA iteration had to have returned a sample-path feasible solution
            // since restarts are repeated until a feasible solution is found.
            if(allCurr[r-1].feas==1 && allInc[r-1].feas==1) {
                
                // No sample path feasible sol obtained so far
                // allBest[r-1]=allInc[r-1]
                // (Doesn't matter what Abest is.)
                for(i=0;i<=id;i++){
                    allBest[r-1].x[i]=allInc[r-1].x[i];
                }
                for(i=0;i<=numConst+1;i++){
                    allBest[r-1].hgmean[i]=allInc[r-1].hgmean[i];
                    for(j=0;j<=numConst+1;j++){
                        allBest[r-1].hgcov[i][j]=allInc[r-1].hgcov[i][j];
                    }
                }
                for(i=0;i<=numSeeds;i++){
                    allBest[r-1].lastseed[i]=allInc[r-1].lastseed[i];
                }
                allBest[r-1].m=allInc[r-1].m;
                allBest[r-1].phat=allInc[r-1].phat;
                allBest[r-1].feas=allInc[r-1].feas;

                
                // Restart rth local seach from a different starting point
                repFlag=1;
                repCount++;
            }
            
            
            // POSSIBILITY II.
            // Previous restart returned a sample-path feasible solution, but the
            // current restart did not.
            else if(allCurr[r-1].feas==1 && allInc[r-1].feas==0) {
                
                // allBest[r-1]=allInc[r-1]
                for(i=0;i<=id;i++){
                    allBest[r-1].x[i]=allInc[r-1].x[i];
                }
                for(i=0;i<=numConst+1;i++){
                    allBest[r-1].hgmean[i]=allInc[r-1].hgmean[i];
                    for(j=0;j<=numConst+1;j++){
                        allBest[r-1].hgcov[i][j]=allInc[r-1].hgcov[i][j];
                    }
                }
                for(i=0;i<=numSeeds;i++){
                    allBest[r-1].lastseed[i]=allInc[r-1].lastseed[i];
                }
                allBest[r-1].m=allInc[r-1].m;
                allBest[r-1].phat=allInc[r-1].phat;
                allBest[r-1].feas=allInc[r-1].feas;
            
                // Restart seach from a different starting point for the same r
                repFlag=1;
                repCount++;
            }
            

            // POSSIBILITY III.
            // *** WILL NEVER OCCUR ***
            // First time a restart returns a sample-path feas. sol.
            // (Will never happen for r>=2 since a restart is repeated until a soln is found.)
            else if(allCurr[r-1].feas==0 && allInc[r-1].feas==1) {
                
                // Abest(r)=Acurr(r);
                for(i=0;i<=id;i++){
                    allBest[r-1].x[i]=allCurr[r-1].x[i];
                }
                for(i=0;i<=numConst+1;i++){
                    allBest[r-1].hgmean[i]=allCurr[r-1].hgmean[i];
                    for(j=0;j<=numConst+1;j++){
                        allBest[r-1].hgcov[i][j]=allCurr[r-1].hgcov[i][j];
                    }
                }
                for(i=0;i<=numSeeds;i++){
                    allBest[r-1].lastseed[i]=allCurr[r-1].lastseed[i];
                }
                allBest[r-1].m=allCurr[r-1].m;
                allBest[r-1].phat=allCurr[r-1].phat;
                allBest[r-1].feas=allCurr[r-1].feas;
                
            }
            
            // POSSIBILITY IV.
            // Restart returns a sample-path feas. sol.
            // Incumbent was also sample-path feasible.
            else {

                if(allCurr[r-1].m>allInc[r-1].m) { // Update the incumbent

                    // Call Oracle at allInc[r-1].x with sample size allCurr[r-1].m
                    // and input seed = allCurr[r-1].lastseed.
                    // Update other variables associated with allInc[r-1].x.

                    xOracle = callOracle(allInc[r-1].x, allCurr[r-1].m, allCurr[r-1].lastseed);
                    for(i=0;i<=numConst+1;i++){
                        allInc[r-1].hgmean[i]=xOracle.hgmean[i];
                        for(j=0;j<=numConst+1;j++){
                            allInc[r-1].hgcov[i][j]=xOracle.hgcov[i][j];
                        }
                    }
                    if(xOracle.flag==0){
                        totWork += allCurr[r-1].m;
                    }
                    allInc[r-1].m=allCurr[r-1].m;
                    allInc[r-1].phat=1;
                    delta[0]=numConst;
                    for(i=2;i<=numConst+1;i++){
                        epsilon[i]=fminl(fmaxl(sqrt(allInc[r-1].hgcov[i][i]*allInc[r-1].m)/pow(allInc[r-1].m,0.45), -allInc[r-1].hgmean[i]), sqrt(allInc[r-1].hgcov[i][i]*allInc[r-1].m)/pow(allInc[r-1].m,0.1));
                        delta[i]=(log(sqrt(allInc[r-1].hgcov[i][i]*allInc[r-1].m))-log(epsilon[i]))/log(allInc[r-1].m);
                        if(isnan(delta[i])){delta[i]=delta0;}
                        if(isinf(1.0/allInc[r-1].hgcov[i][i])){
                            allInc[r-1].phat=0.0;
                        }else{
                            allInc[r-1].phat = allInc[r-1].phat * gsl_cdf_tdist_P ( pow(allInc[r-1].m,(0.5-delta[i]))- allInc[r-1].hgmean[i]/sqrt(allInc[r-1].hgcov[i][i]), allInc[r-1].m-1 );
                        }
                    }
                    if(xOracle.flag==0) allInc[r-1].feas=0;
                    else allInc[r-1].feas=1;
                        
                }
                
                
                if(allCurr[r-1].m<allInc[r-1].m) { // Update current solution
                    
                    // Call Oracle at allCurr[r-1].x with sample size allInc[r-1].m
                    // and input seed = allInc[r-1].lastseed.
                    // Update other variables associated with allCurr[r-1].x.
                    
                    xOracle = callOracle(allCurr[r-1].x, allInc[r-1].m, allInc[r-1].lastseed);
                    for(i=0;i<=numConst+1;i++){
                        allCurr[r-1].hgmean[i]=xOracle.hgmean[i];
                        for(j=0;j<=numConst+1;j++){
                            allCurr[r-1].hgcov[i][j]=xOracle.hgcov[i][j];
                        }
                    }
                    if(xOracle.flag==0){
                        totWork += allInc[r-1].m;
                    }
                    allCurr[r-1].m=allInc[r-1].m;
                    allCurr[r-1].phat=1;
                    delta[0]=numConst;
                    for(i=2;i<=numConst+1;i++){
                        epsilon[i]=fminl(fmaxl(sqrt(allCurr[r-1].hgcov[i][i]*allCurr[r-1].m)/pow(allCurr[r-1].m,0.45), -allCurr[r-1].hgmean[i]), sqrt(allCurr[r-1].hgcov[i][i]*allCurr[r-1].m)/pow(allCurr[r-1].m,0.1));
                        delta[i]=(log(sqrt(allCurr[r-1].hgcov[i][i]*allCurr[r-1].m))-log(epsilon[i]))/log(allCurr[r-1].m);
                        if(isnan(delta[i])){delta[i]=delta0;}
                        if(isinf(1.0/allCurr[r-1].hgcov[i][i])){
                            allCurr[r-1].phat=0.0;
                        }else{
                            allCurr[r-1].phat = allCurr[r-1].phat * gsl_cdf_tdist_P ( pow(allCurr[r-1].m,(0.5-delta[i]))- allCurr[r-1].hgmean[i]/sqrt(allCurr[r-1].hgcov[i][i]), allCurr[r-1].m-1 );
                        }
                    }
                    if(xOracle.flag==0) allCurr[r-1].feas=0;
                    else allCurr[r-1].feas=1;

                }
            
                
                // Update allBest:
                
                //After updating incumbent, incumbent is sample-path infeasible
                // or is not alphar-feasible.
                if(allInc[r-1].feas==1 || allInc[r-1].phat < alphar) inc_good=0;
                else inc_good=1;

                //After updating incumbent, incumbent is sample-path infeasible
                // or is not alphar-feasible.
                if(allCurr[r-1].feas==1 || allCurr[r-1].phat < alphar) curr_good=0;
                else curr_good=1;

                // Notice that inc_good and curr_good can never be 0 at the same
                // time since only one of the two is ever updated. The other is
                // always sample-path as well as alphar feasible
                
                
                if(inc_good == 1 && curr_good == 0) {
                    
                    // allBest[r-1]=allInc[r-1]
                    for(i=0;i<=id;i++){
                        allBest[r-1].x[i]=allInc[r-1].x[i];
                    }
                    for(i=0;i<=numConst+1;i++){
                        allBest[r-1].hgmean[i]=allInc[r-1].hgmean[i];
                        for(j=0;j<=numConst+1;j++){
                            allBest[r-1].hgcov[i][j]=allInc[r-1].hgcov[i][j];
                        }
                    }
                    for(i=0;i<=numSeeds;i++){
                        allBest[r-1].lastseed[i]=allInc[r-1].lastseed[i];
                    }
                    allBest[r-1].m=allInc[r-1].m;
                    allBest[r-1].phat=allInc[r-1].phat;
                    allBest[r-1].feas=allInc[r-1].feas;

                    
                }
                else if(inc_good == 0 && curr_good == 1) {
                    
                    // allBest[r-1]=allCurr[r-1]
                    for(i=0;i<=id;i++){
                        allBest[r-1].x[i]=allCurr[r-1].x[i];
                    }
                    for(i=0;i<=numConst+1;i++){
                        allBest[r-1].hgmean[i]=allCurr[r-1].hgmean[i];
                        for(j=0;j<=numConst+1;j++){
                            allBest[r-1].hgcov[i][j]=allCurr[r-1].hgcov[i][j];
                        }
                    }
                    for(i=0;i<=numSeeds;i++){
                        allBest[r-1].lastseed[i]=allCurr[r-1].lastseed[i];
                    }
                    allBest[r-1].m=allCurr[r-1].m;
                    allBest[r-1].phat=allCurr[r-1].phat;
                    allBest[r-1].feas=allCurr[r-1].feas;
                    
                    
                }
                else if(inc_good == 0 && curr_good == 0){ // not possible!
                    
                    // Error: By default set allBest[r-1]=allInc[r-1]
                    for(i=0;i<=id;i++){
                        allBest[r-1].x[i]=allInc[r-1].x[i];
                    }
                    for(i=0;i<=numConst+1;i++){
                        allBest[r-1].hgmean[i]=allInc[r-1].hgmean[i];
                        for(j=0;j<=numConst+1;j++){
                            allBest[r-1].hgcov[i][j]=allInc[r-1].hgcov[i][j];
                        }
                    }
                    for(i=0;i<=numSeeds;i++){
                        allBest[r-1].lastseed[i]=allInc[r-1].lastseed[i];
                    }
                    allBest[r-1].m=allInc[r-1].m;
                    allBest[r-1].phat=allInc[r-1].phat;
                    allBest[r-1].feas=allInc[r-1].feas;

                }
                else {
                    if(allCurr[r-1].hgmean[1]<allInc[r-1].hgmean[1]) {
                        
                        // allBest[r-1]=allCurr[r-1]
                        for(i=0;i<=id;i++){
                            allBest[r-1].x[i]=allCurr[r-1].x[i];
                        }
                        for(i=0;i<=numConst+1;i++){
                            allBest[r-1].hgmean[i]=allCurr[r-1].hgmean[i];
                            for(j=0;j<=numConst+1;j++){
                                allBest[r-1].hgcov[i][j]=allCurr[r-1].hgcov[i][j];
                            }
                        }
                        for(i=0;i<=numSeeds;i++){
                            allBest[r-1].lastseed[i]=allCurr[r-1].lastseed[i];
                        }
                        allBest[r-1].m=allCurr[r-1].m;
                        allBest[r-1].phat=allCurr[r-1].phat;
                        allBest[r-1].feas=allCurr[r-1].feas;
                        
                        
                    }
                    else {
                        
                        // allBest[r-1]=allInc[r-1]
                        for(i=0;i<=id;i++){
                            allBest[r-1].x[i]=allInc[r-1].x[i];
                        }
                        for(i=0;i<=numConst+1;i++){
                            allBest[r-1].hgmean[i]=allInc[r-1].hgmean[i];
                            for(j=0;j<=numConst+1;j++){
                                allBest[r-1].hgcov[i][j]=allInc[r-1].hgcov[i][j];
                            }
                        }
                        for(i = 0; i <= numSeeds; i++) {
                            allBest[r-1].lastseed[i] = allInc[r-1].lastseed[i];
                        }
                        allBest[r-1].m=allInc[r-1].m;
                        allBest[r-1].phat=allInc[r-1].phat;
                        allBest[r-1].feas=allInc[r-1].feas;
                        
                    }
                }
            }
        }
        
        fprintf(logFile,"r=%d, total budget expended = %d\n", r, totWork);
        printf("r=%d, total budget expended = %d\n", r, totWork);

        // Increment r if restart returns a sample-path feasible solution,
        // else restart rth local search with a new seed
        if(repFlag == 0) {
            // write bestSols
            fwrite (&r, sizeof(r), 1, bestSols);
            fwrite (&br, sizeof(br), 1, bestSols);
            fwrite (&totWork, sizeof(totWork), 1, bestSols);
            fwrite (&alphar, sizeof(alphar), 1, bestSols);
            fwrite (&allBest[r-1], sizeof(allBest[r-1]), 1, bestSols);
            
//            // write currSols
//            fwrite (&r, sizeof(r), 1, currSols);
//            fwrite (&br, sizeof(br), 1, currSols);
//            fwrite (&totWork, sizeof(totWork), 1, currSols);
//            fwrite (&alphar, sizeof(alphar), 1, currSols);
//            fwrite (&allCurr[r-1], sizeof(allCurr[r-1]), 1, currSols);
//
//            
//            // write incSols
//            fwrite (&r, sizeof(r), 1, incSols);
//            fwrite (&br, sizeof(br), 1, incSols);
//            fwrite (&totWork, sizeof(totWork), 1, incSols);
//            fwrite (&alphar, sizeof(alphar), 1, incSols);
//            fwrite (&allInc[r-1], sizeof(allInc[r-1]), 1, incSols);

            r++;
            repCount=1;
        }
        

    }
    
    toc = clock();
    timeElapsed=(float)(toc - tic) / (60*CLOCKS_PER_SEC);
    fprintf(logFile, "Time elapsed: %f minutes\n", (float)(toc - tic) / (60*CLOCKS_PER_SEC));
    fprintf(logFile, "Total budget expended = %d\n", totWork);
    printf("Time elapsed: %f minutes\n", (float)(toc - tic) / (60*CLOCKS_PER_SEC));
    printf("Total budget expended = %d\n", totWork);

    fwrite (&tFlag, sizeof(timeElapsed), 1, allSols);
    fwrite (&timeElapsed, sizeof(timeElapsed), 1, allSols);
    fclose(allSols);
    fclose(bestSols);
//    fclose(incSols);
//    fclose(currSols);
    fclose(logFile);
    return;
    
}

//// main: ===============================================
//int main()
//{
//    int budg;
//    int probSeed[10], solvSeed[10], xmin[10], xmax[10];
//    
//    // SET PROBLEM PARAMETERS
//    probParam[0]  = 1;  //probID
//    probParam[1]  = 2;  //probDim = 2
//    probParam[2]  = 1;  //numSeed = 1
//    probParam[3]  = 1;  //numConst = 1
//    probParam[4]  = 25; //lambda = 25 (demand intensity)
//    probParam[5]  = 32; //A = 32 (fixed order cost)
//    probParam[6]  = 3;  //C = 3 (unit order cost )
//    probParam[7]  = 1;  //I = 1 (inventory holding cost)
//    probParam[8]  = 5;  //PI = 5 (unit backorder cost)
//    probParam[9]  = 100; //warmup = 100
//    probParam[10] = 30; //simLength = 30
//    
//    // SET SOLVER PARAMETERS
//    solvParam[0]  = 1;     // num. final solutions
//    solvParam[1]  = 100;   // num. restarts
//    solvParam[2]  = 1000;  // kmax
//    solvParam[3]  = 3.5;   // q
//    solvParam[4]  = 0.4;   // delta
//    solvParam[5]  = 10;    // c1
//    solvParam[6]  = 1.1;   // c2
//    solvParam[7]  = 0.002; // cvthreshold
//    solvParam[8]  = 0.2;   // biasthreshold
//    solvParam[9]  = 1;     // yesgeom for mk
//    solvParam[10] = 8;     // a in br=a*c^k
//    solvParam[11] = 1.1;   // c in br=a*c^k
//    solvParam[12] = 500;   // restart budget constant, a
//    solvParam[13] = 1.1;   // restart budget exponent, c (br = a*r^c)
//    solvParam[14] = 0;     // alpha in (alpha_r = alpha*(1-c^(r+1)))
//    solvParam[15] = 0.65;  // c in (alpha_r = alpha*(1-c^(r+1)))
//    solvParam[16] = 0;     // yesgeom for b_r
//    solvParam[17] = 0;     // ftol (function tolerance)
//    solvParam[18] = 1;     // q-quantile in pi(q,mk)
//    
//    probSeed[0] = 1; //size of probSeed = 1
//    probSeed[1] = 825;
//    probSeed[2] = 64839;
//    
//    solvSeed[0] = 2; //size of solvSeed = 2
//    solvSeed[1] = 4;
//    solvSeed[2] = 6;
//    
//    xmin[0]=2;
//    xmax[0]=2;
//    
//    xmin[1]=20;
//    xmax[1]=46;
//    
//    xmin[2]=40;
//    xmax[2]=100;
//    
//    budg=10000;
//    
//    strcpy(logPath,"/Users/kalyani/0_CODE/cgRSPLINE_xcode/");
//
//    // CALL cgR-SPLINE
//    cgRSPLINE(xmin, xmax, probSeed, solvSeed, budg);
//
//    return 0;
//    
//}


int main(int argc, const char * argv[]) {
    int i, j, probDim;
    int budg, probSeed[10], solvSeed[10], xmin[10], xmax[10];
    
    // Define default parameters:
    solvParam[0]  = 1;     // num. final solutions
    solvParam[1]  = 100;   // max. restarts
    solvParam[2]  = 1000;  // max. RA iterations
    solvParam[3]  = 0.4;   // delta0
    solvParam[4]  = 1;     // yesGeom for mk
    solvParam[5]  = 8;     // mk0
    solvParam[6]  = 10;    // bk0
    solvParam[7]  = 1.1;   // c1 (a1 if yesGeom==1, else q1)
    solvParam[8]  = 0;     // yesGeom for br
    solvParam[9]  = 500;   // b0
    solvParam[10] = 1.1;   // c2 (a2 if yesGeom==1, else q2)
    solvParam[11] = 0;     // alpha  (alphar = alpha*(1-alpah0^(r+1)))
    solvParam[12] = 0.65;  // alpha0 (alphar = alpha*(1-alpah0^(r+1)))
    solvParam[13] = 1;     // p-quantile in pi(p,mk)
    solvParam[14] = 0;     // ftol (function tolerance)
    
//    // For the inventory problem
//    probParam[0]  = 1;  // probID = 1
//    probParam[1]  = 2;  // probDim = 2
//    probParam[2]  = 1;  // numSeed = 1
//    probParam[3]  = 1;  // numConst = 1
//    probParam[4]  = 25; // lambda = 25 (demand intensity)
//    probParam[5]  = 32; // A = 32 (fixed order cost)
//    probParam[6]  = 3;  // C = 3 (unit order cost )
//    probParam[7]  = 1;  // I = 1 (inventory holding cost)
//    probParam[8]  = 5;  // PI = 5 (unit backorder cost)
//    probParam[9]  = 100;// warmUp = 100
//    probParam[10] = 30; // simLength = 30
//    
//    probID       = 1;
//    probDim      = 2;
//
//    probSeed[0] = 1; //size of probSeed = 1
//    probSeed[1] = 825;
//    probSeed[2] = 64839;
//    solvSeed[0] = 2; //size of solvSeed = 2
//    solvSeed[1] = 4;
//    solvSeed[2] = 6;
//    
//    xmin[0]=2;
//    xmax[0]=2;
//    xmin[1]=20;
//    xmax[1]=46;
//    xmin[2]=40;
//    xmax[2]=100;
    
    // For the three-stage flowline problem
    probParam[0]  = 2;  // probID = 1
    probParam[1]  = 4;  // probDim = 2
    probParam[2]  = 3;  // numSeed = 1
    probParam[3]  = 1;  // numConst = 1
    probParam[4]  = 50; // warmup time = 100
    probParam[5]  = 1000;// simulation end time = 130
    probParam[6]  = 20; // total service rate = 20
    probParam[7]  = 20; // total buffer space available = 20
    
    probID       = 2;
    probDim      = 4;
    
    probSeed[0] = 3; //size of probSeed = 2
    probSeed[1] = 88294;
    probSeed[2] = 82057;
    probSeed[3] = 25;
    
    solvSeed[0] = 4; //size of solvSeed = 4
    solvSeed[1] = 9375;
    solvSeed[2] = 72;
    solvSeed[3] = 1;
    solvSeed[4] = 84728864;
    
    xmin[0]=4;
    xmax[0]=4;
    xmin[1]=5;
    xmax[1]=20;
    xmin[2]=5;
    xmax[2]=20;
    xmin[3]=5;
    xmax[3]=20;
    xmin[4]=5;
    xmax[4]=19;
    
    // Other parameters
    budg=10000;
    strcpy(logPath,"/Users/kalyani/0_CODE/Xcode/cgRSPLINE/");
    strcpy(expName, "test");
    
    // Read optional parameters
    for(i = 1; i < argc ;i++) {
        
        // First parse problem-independent parameters
        if (strcmp(argv[i], "-expName") == 0) //expName (str)
        {
            strcpy(expName, argv[i+1]); // budg (int)
        }
        if (strcmp(argv[i], "-budg") == 0)
        {
            budg = atoi(argv[i+1]); // budg (int)
        }
        if (strcmp(argv[i], "-alpha") == 0)
        {
            solvParam[11] = atof(argv[i+1]); // alpha (float)
        }
        if (strcmp(argv[i], "-pi") == 0)
        {
            solvParam[13] = atof(argv[i+1]); // p-quantile of pi (float)
        }
        if (strcmp(argv[i], "-b") == 0)
        {
            solvParam[9] = atoi(argv[i+1]); // b0 (int)
        }
        if (strcmp(argv[i], "-a1") == 0)
        {
            solvParam[8] = 1;
            solvParam[7] = atof(argv[i+1]); // a1 (float)
        }
        if (strcmp(argv[i], "-q1") == 0)
        {
            solvParam[8] = 0;
            solvParam[7] = atof(argv[i+1]); // q1 (float)
        }
        if (strcmp(argv[i], "-a2") == 0)
        {
            solvParam[4] = 1;
            solvParam[10] = atof(argv[i+1]); // a2 (float)
        }
        if (strcmp(argv[i], "-q2") == 0)
        {
            solvParam[4] = 0;
            solvParam[10] = atof(argv[i+1]); // q2 (float)
        }
        
        
        // Now parse problem name
        if (strcmp(argv[i], "sSINV") == 0)
        {
            probID       = 1;
            probDim      = 2;
            probSeed[0]  = 1;
            solvSeed[0]  = 1;
        }
        if (strcmp(argv[i], "TSF") == 0)
        {
            probID       = 2;
            probDim      = 4;
            probSeed[0]  = 3;
            solvSeed[0]  = 4;
        }
        
    }
    
    // Second pass
    for(i = 1; i < argc ;i++) {
        
        // problem seed
        if (strcmp(argv[i], "-ps") == 0)
        {
            for (j = 1; j <= probSeed[0]; j++ )
            probSeed[j] = atoi(argv[i+j]);
        }
        // solver seed
        if (strcmp(argv[i], "-ss") == 0)
        {
            for (j = 1; j <= solvSeed[0]; j++ )
                solvSeed[j] = atoi(argv[i+j]);
        }
        // xmin
        if (strcmp(argv[i], "-xmin") == 0)
        {
            for (j = 1; j <= probDim; j++ )
                xmin[j] = atoi(argv[i+j]);
        }
        // xmax
        if (strcmp(argv[i], "-xmax") == 0)
        {
            for (j = 1; j <= probDim; j++ )
                xmax[j] = atoi(argv[i+j]);
        }
        
    }
    
    // CALL cgR-SPLINE
    cgRSPLINE(xmin, xmax, probSeed, solvSeed, budg);

    
    return 0;
}


