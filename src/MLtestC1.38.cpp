//g++ -o MLtestC#.# MLtestC#.#.cpp glut32.lib -lopengl32 -lglu32 -lopenal32 -static-libgcc -static-libstdc++  //use this format to compile
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <windows.h>
#include "GL/glut.h"
#include <cmath>
#include <vector>
using namespace std;

#define BUILDID "1.30"
/*Changes from 1.351:
    1.8 with:
        - Action corresponds to acceleration rather than displacement. This is more realistic for robotic applications.
        - Cart velocity added as input.
*/

#define TWOPI 6.28319
#define INF 999999999999999

/*
//Simulation parameters
#define INPUTNUM (2*RAYNUM+1)
#define ACTIONNUM 2

#define ROOMVERTICES 10
#define VELOCITY .2*BOTSIZE
#define ANGVELOCITY 2*VELOCITY
#define SIGHTRANGE 10*BOTSIZE
#define PADSIZE BOTSIZE/2.0
#define BOTSIZE .2
#define SIGHTFOV (TWOPI/4)
#define RAYNUM 3
#define STANDARDREWARD 1
#define REWARDBIAS (-STANDARDREWARD/100.0)
#define AINIT 0
#define GAMMA .8
#define TIMEPERUPDATE 1 //must be a natural number
#define TEMPERATURE 0
#define P_USEMAX 0.9

#define NNDEPTH 2 //The number of neuronal layers in network. ==Number of weight layers. ==(Number of layers in "input" variable)-1. ==(Number of hidden layers)+1
#define NNLEARNINGRATE1 .00001
#define FRICTIONTAU 10 //step decays by exp(-1/RELAXSTEPDECAY) every timestep
#define NNWIDTH INPUTNUM+ACTIONNUM

#define STATESTR "ray0 r\tray1 r...\tray0 color\tray1 color...\ttouching?"
#define ACTIONSTR "translation\trotation"
*/

//Simulation parameters
#define GAMMA .5
#define TIMEPERUPDATE 1 //k*DeltaT in Advantage learning algorithm
#define INPUTNUM 3 //Number of NN inputs
#define ACTIONNUM 3 //Number of NN outputs
#define NNDEPTH 5 //The number of neuronal layers in network. ==Number of weight layers. ==(Number of layers in "input" variable)-1
#define NNLEARNINGRATE1DECAYTAU INF
#define NNWIDTH INPUTNUM+ACTIONNUM //Max width of NN
#define NNLEARNINGRATE1 .001

#define RODLENGTH 1
#define RESETANGLE .2
#define GRAVITY .2
#define CARTFORCE .02
#define CARTFRICTION .7
#define DRAWSIZE .1
#define PLOTWIDTH 200
#define RESETCOUNTLIMIT 2000
#define VTERMINAL 0.2
#define KADV 2
#define RLTYPE 3 //See updateV() for more definitions

#define EPSILON1 1
#define EPSILON2 .1
#define EPSILONDECAYTIME 10

//Replay parameters
#define REPLAYNUM 40 //Number of replays per frame
#define MEMSIZE 10000 //For no replay, set MEMSIZE = 2 so that it only stores current and next state.
#define REPLAYSKIPTHRESH .05 //VdiffMem value at which there is 50% probability of skipping (0 to never skip)

#include "NNtoolsC.h"

GLint winw=700,winh=600;
GLfloat viewradius=2.5,pbot[3]={-.8,.3,0},pcharger[2]={1,0},room[10][2];

int nnwidth[NNDEPTH+1],simtime=0,logtimestamp,nextmemoryi=0,resetcount=0;
float epsilon,rodangle,rodvelocity,cartpos,cartvel,alpha,walldistance,input[NNDEPTH+1][NNWIDTH],weight[NNDEPTH][NNWIDTH][NNWIDTH],bias[NNDEPTH][NNWIDTH],weightMSgrad[NNDEPTH][NNWIDTH][NNWIDTH],biasMSgrad[NNDEPTH][NNWIDTH];
float Vcur,stateMem[MEMSIZE][INPUTNUM],rewardMem[MEMSIZE],VdiffMem[MEMSIZE],terminalMem[MEMSIZE];
float history[4][PLOTWIDTH],Ravg[RESETCOUNTLIMIT],actionmap[3]={-1,0,1},viewscales=2;
int historyindex=0,memindex=0,episodelength,actionMem[MEMSIZE];
bool dispbit=true,pausebit=false,chargergrabbed=false,stuck=false;

void resetproblem(){
    rodangle=RESETANGLE*pow(-1,rand());
    rodvelocity=0;
    cartpos=randuniform(-RODLENGTH/2.0,RODLENGTH/2.0);
    cartvel=0;
    Ravg[resetcount]/=episodelength;
    resetcount++;
    Ravg[resetcount]=0;
    episodelength=1;
    printf("\b\b\b\b\b\b%06i",resetcount);
}

void resetall(){
    epsilon=EPSILON1;
    
    resetproblem();
    resetcount=0;
    RandomizeNN(nnwidth,NNDEPTH,weight,bias,1.0/sqrt(NNWIDTH));
    InitNN(nnwidth,NNDEPTH,weightMSgrad,biasMSgrad,1);
    
    simtime=0;
    memindex=0;
    historyindex=0;
}

/*
bool checkline(float* p0,float* p1,float* p2,float* intersection){
    //If triangle formed by points has an obtuse angle, then point is outside of the ends of the line segment
    if(((p1[0]-p0[0])*(p1[0]-p2[0])+(p1[1]-p0[1])*(p1[1]-p2[1]))*((p2[0]-p0[0])*(p2[0]-p1[0])+(p2[1]-p0[1])*(p2[1]-p1[1]))<0)return false;
    //Otherwise, return the point on the line formed by p1p2 that is closest to p0
    float m=(p2[1]-p1[1])/(p2[0]-p1[0]),b=p1[1]-m*p1[0];
    intersection[0]=(p0[0]+m*p0[1]-m*b)/(m*m+1);
    intersection[1]=(m*(p0[0]+m*p0[1])+b)/(m*m+1);
    return true;
}
bool checkstuck(float* p){
    int i;
    float intersection[2];
    //if(dispbit || t%200==0)glClear(GL_COLOR_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    
    //Check that every wall forms a counter-clockwise triangle with the point.
    for(i=0;i<ROOMVERTICES;i++){
        if(checkline(p,room[i],room[(i+1)%ROOMVERTICES],intersection)){
            
            //Display wall intersections where applicable.
            /*if(dispbit || t%200==0){
                glViewport(0,0,winw,winh);
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                gluOrtho2D(-viewradius*winw/winh,viewradius*winw/winh,viewradius,-viewradius);
                
                glPointSize(6);
                glColor3f(0,0,1);
                glBegin(GL_POINTS);
                    glVertex2fv(intersection);
                glEnd();
            }*/
            /*
            if((room[i][0]-p[0])*(room[i][1]+p[1])+(room[i+1][0]-room[i][0])*(room[i+1][1]+room[i][1])+(p[0]-room[i+1][0])*(p[1]+room[i+1][1])>0)return true;
            else if(sqrt(pow(p[0]-intersection[0],2)+pow(p[1]-intersection[1],2))<=BOTSIZE*sqrt(2))return true;
        }
    }
    return false;
}
*/

float min(float n1,float n2){
    if(n1<n2)return n1;
    else return n2;
}
float minv(float* v,int size){
    if(size==0)return 0;
    int i,imin=0;
    for(i=1;i<size;i++)if(v[i]<v[imin])imin=i;
    return v[imin];
}
float max(float n1,float n2){
    if(n1>n2)return n1;
    else return n2;
}
float maxv(float* v,int size){
    if(size==0)return 0;
    int i,imax=0;
    for(i=1;i<size;i++)if(v[i]>v[imax])imax=i;
    return v[imax];
}
int maxvi(float* v,int size){
    if(size==0)return 0;
    int i,imax=0;
    for(i=1;i<size;i++)if(v[i]>v[imax])imax=i;
    return imax;
}    

float* interpV(float *statev,float *actionv){
    int i;
    for(i=0;i<INPUTNUM;i++)input[0][i]=statev[i];
    ApplyNN(nnwidth,NNDEPTH,weight,bias,&acttanh,input);
    for(i=0;i<ACTIONNUM;i++)actionv[i]=input[NNDEPTH][i];
    return actionv;
}

float* stateID(float* stateptr){
    //Normalized inputs
    stateptr[0]=rodangle/(TWOPI/4);
    stateptr[1]=2*cartpos/RODLENGTH;
    stateptr[2]=cartvel/(2*CARTFORCE);
    //stateptr[2]=rodvelocity/(GRAVITY+CARTVELOCITY);
    
    /*
    int i,wall,inputval[RAYNUM+1];
    float ray=0,r,j,distance,intersection[2];
    
    stateptr[2*RAYNUM]=0;
    //Cycle through visual rays
    for(i=0,ray=-1;i<RAYNUM,ray<=1;i++,ray+=2.0/(RAYNUM-1)){
        stateptr[i]=0;
        stateptr[RAYNUM+i]=0;
        //For each ray, step along the length
        for(r=BOTSIZE;r<SIGHTRANGE;r+=r*SIGHTFOV/((RAYNUM-1)*2.0)){
            //At each point along ray, check for intersection with charger
            float raypoint[2]={pbot[0]+r*cos(pbot[2]+ray*SIGHTFOV/2.0),pbot[1]+r*sin(pbot[2]+ray*SIGHTFOV/2.0)};
            distance=sqrt(pow(raypoint[0]-pcharger[0],2)+pow(raypoint[1]-pcharger[1],2));
            if(distance-PADSIZE<=r*SIGHTFOV/((RAYNUM-1)*2.0)){
                stateptr[i]=acttanh(r,0);
                stateptr[RAYNUM+i]=2;
                distance=sqrt(pow(pbot[0]-pcharger[0],2)+pow(pbot[1]-pcharger[1],2));
                if(distance<BOTSIZE+PADSIZE){
                    stateptr[2*RAYNUM]=1;
                    if(ray==0)reward+=STANDARDREWARD;
                }
                break;
            }
            //At each point along ray, check for intersection with wall
            for(wall=0;wall<ROOMVERTICES;wall++){
                if(!checkline(raypoint,room[wall],room[(wall+1)%ROOMVERTICES],intersection))continue;
                distance=sqrt(pow(raypoint[0]-intersection[0],2)+pow(raypoint[1]-intersection[1],2));
                if(distance<=r*SIGHTFOV/((RAYNUM-1)*2.0)){
                    stateptr[i]=acttanh(r,0);
                    stateptr[RAYNUM+i]=1;
                    break;
                }
            }
            if(inputval[i]>0)break;
        }
    }
    //Check for wall touch
    if(stateptr[2*RAYNUM]==0)if(checkstuck(pbot))stateptr[2*RAYNUM]=1;
    */
    return stateptr;
}

void chooseaction(){
    //reward=REWARDBIAS;
    
    //Choose the most valuable action with probably USEMAX. If not, discard action and repeat.
    /*int quantity=ACTIONNUM,actionlist[ACTIONNUM],i,j,maxi,temp;
    for(i=0;i<quantity;i++)actionlist[i]=i;
    while(quantity>1){
        maxi=0;
        for(i=1;i<quantity;i++)
            if(A[state][actionlist[i]]>A[state][actionlist[maxi]])maxi=i;
        if((rand()%10000)/10000.0<P_USEMAX)return maxi;
        else{
            i=j=0;
            while(i<quantity){
                actionlist[j]=actionlist[i];
                if(i==maxi)j--;
                i++;
                j++;
            }
            quantity--;
        }
    }
    return maxi;*/
    
    //Use temperature to choose an action
    int i;
    float tempaction[ACTIONNUM];
    actionMem[memindex]=maxvi(interpV(stateMem[memindex],tempaction),ACTIONNUM);
    if(randuniform(0,1)<epsilon)actionMem[memindex]=rand()%ACTIONNUM;
}

float* vactioncalc(int replayindex,int RLtype,float *Vaction){
    int i;
    float Vnextaction[ACTIONNUM];
    interpV(stateMem[(replayindex+1)%(MEMSIZE+1)],Vnextaction);
    interpV(stateMem[replayindex],Vaction);
    float Vnext,Vcurmax=maxv(Vaction,ACTIONNUM);
    
    switch(RLtype){
        case 0: //Q-Learning
            if(terminalMem[replayindex])Vnext=Vcur-VTERMINAL;
            else Vnext=maxv(Vnextaction,ACTIONNUM);
            Vaction[actionMem[replayindex]]=rewardMem[replayindex]+GAMMA*Vnext;//-Vaction[actionMem[replayindex]]
            break;
        case 1: //Advantage Learning
            if(terminalMem[replayindex])Vnext=Vcur-VTERMINAL;
            else Vnext=maxv(Vnextaction,ACTIONNUM);
            Vaction[actionMem[replayindex]]=KADV*(rewardMem[replayindex]+pow(GAMMA,1.0/KADV)*Vnext-Vcurmax)+Vcurmax;
            break;
        case 2: //SARSA
            if(terminalMem[replayindex])Vnext=Vcur-VTERMINAL;
            else Vnext=Vnextaction[actionMem[(replayindex+1)%(MEMSIZE+1)]];
            Vaction[actionMem[replayindex]]=rewardMem[replayindex]+GAMMA*Vnext;
            break;
        case 3: //SARSA Advantage (needs update based on case 2)
            if(terminalMem[replayindex])Vnext=Vcur-VTERMINAL;
            else Vnext=Vnextaction[actionMem[(replayindex+1)%(MEMSIZE+1)]];
            Vaction[actionMem[replayindex]]=KADV*(rewardMem[replayindex]+pow(GAMMA,1.0/KADV)*Vnext-Vcurmax)+Vcurmax;
            break;
        case 4: //Temporal Difference Learning
            break;
    }
    return Vaction;
}

void updateV(int replayindex,int RLtype){
    float Vaction[ACTIONNUM];
    vactioncalc(replayindex,RLTYPE,Vaction);
    RMSProp(nnwidth,NNDEPTH,weight,bias,weightMSgrad,biasMSgrad,&acttanh,input,Vaction,NNLEARNINGRATE1,0,NULL);
    if(CheckNN(nnwidth,NNDEPTH,weight,bias,1000)){
        cout<<"\tError: NN diverging. Forcing reset.\n";
        resetall();
    }
}

void applyaction(){
    cartvel+=CARTFORCE*actionmap[actionMem[memindex]];
    cartpos+=cartvel;
    float rodchange=-cartvel*cos(rodangle); //Effect of the cart on the rod
    rodchange+=GRAVITY*sin(rodangle)*abs(sin(rodangle)); //Effect of gravity on the rod
    rodangle+=rodchange;
    //rodvelocity=GRAVITY*sin(rodangle)*abs(sin(rodangle))-CARTVELOCITY*actionMem[memindex][0]*cos(rodangle); //Angular velocity of rod
    rodangle+=randuniform(-.001,.001); //To keep the learner on its toes.
    float actionchange=actionmap[actionMem[memindex]]-actionmap[actionMem[(memindex+MEMSIZE-1)%MEMSIZE]];
    
    rewardMem[memindex]=-1*abs(rodangle)-.5*abs(cartpos)/RODLENGTH;//-10*abs(cartvel);//-.2*abs(actionchange); //Reward based on absolute angle, cart displacement, and effort
    cartvel*=CARTFRICTION;
    
    if(abs(cartpos)>RODLENGTH || abs(rodangle)>TWOPI/4.0){
        //rewardMem[memindex]-=2; //Terminal punishment
        terminalMem[memindex]=1;
    }
    else terminalMem[memindex]=0;
    Ravg[resetcount]+=abs(rodangle);
    episodelength++;
    
    /*
    pbot[2]+=ANGVELOCITY*tanh(action[1]);
    
    GLfloat newpbot[3];
    newpbot[0]=pbot[0]+2*tanh(action[0])*VELOCITY*cos(pbot[2]);
    newpbot[1]=pbot[1]+2*tanh(action[0])*VELOCITY*sin(pbot[2]);
    
    //reward-=abs(2*action[nextmemoryi][0]*VELOCITY*cos(pbot[2]));
    //reward-=abs(action[nextmemoryi][1]);
    
    if(checkstuck(newpbot)){
        stuck=true;
        reward-=STANDARDREWARD;
    }
    else{
        stuck=false;
        pbot[0]=newpbot[0];
        pbot[1]=newpbot[1];
    }
    */
}
        
void disp(){
    int i;
    float Vaction[ACTIONNUM];
    interpV(stateMem[memindex],Vaction);
    
	if(dispbit || simtime%50==0){
        float j;
        glClear(GL_COLOR_BUFFER_BIT);		     // Clear Screen and Depth Buffer
        
        //This is necessary for zoom updates.
    	glViewport(0,3*winh/4.0,winw/2,winh/4.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(-(DRAWSIZE+.6*RODLENGTH)*(winw/2.0)/(winh/4.0),(DRAWSIZE+.6*RODLENGTH)*(winw/2.0)/(winh/4.0),-2.0*DRAWSIZE,1.2*RODLENGTH);
        glLineWidth(2);
        glColor3f(1,1,1);
        glBegin(GL_LINE_STRIP);
            glVertex2f(-RODLENGTH,-DRAWSIZE);
            glVertex2f(-RODLENGTH,1.1*RODLENGTH);
            glVertex2f(+RODLENGTH,1.1*RODLENGTH);
            glVertex2f(+RODLENGTH,-DRAWSIZE);
            glVertex2f(-RODLENGTH,-DRAWSIZE);
        glEnd();
            
        
        /*
        //Disply room
            glColor3f(1,0,0);
            glBegin(GL_LINE_LOOP);
            	for(i=0;i<ROOMVERTICES;i++)glVertex2f(room[i][0],room[i][1]);
            glEnd();
        //Disply bot
            glColor3f(1,1,(float)(!stuck));
            glBegin(GL_QUADS);
                for(i=0;i<=4;i++)glVertex2f(pbot[0]+BOTSIZE*sqrt(2)*cos(pbot[2]+TWOPI*(0.125+0.25*i)),pbot[1]+BOTSIZE*sqrt(2)*sin(pbot[2]+TWOPI*(0.125+0.25*i)));
            glEnd();
            glColor3f(.5,.5,.5);
            //Display rays
            glBegin(GL_LINES);
                for(j=-1;j<=1;j+=2.0/(RAYNUM-1)){
                    glVertex2f(pbot[0],pbot[1]);
                    glVertex2f(pbot[0]+SIGHTRANGE*cos(pbot[2]+j*SIGHTFOV/2.0),pbot[1]+SIGHTRANGE*sin(pbot[2]+j*SIGHTFOV/2.0));
                }
            glEnd();
        //Display charger
            glColor3f(0,1,0);
            glBegin(GL_QUADS);
                glVertex2f(pcharger[0]+PADSIZE,pcharger[1]+PADSIZE);
                glVertex2f(pcharger[0]-PADSIZE,pcharger[1]+PADSIZE);
                glVertex2f(pcharger[0]-PADSIZE,pcharger[1]-PADSIZE);
                glVertex2f(pcharger[0]+PADSIZE,pcharger[1]-PADSIZE);
            glEnd();
        */
        
        //Disply cart
            glColor3f(.5,.5,.5);
            glBegin(GL_QUADS);
                glVertex2f(cartpos+DRAWSIZE,0);
                glVertex2f(cartpos+DRAWSIZE,-DRAWSIZE);
                glVertex2f(cartpos-DRAWSIZE,-DRAWSIZE);
                glVertex2f(cartpos-DRAWSIZE,0);
            glEnd();
        //Disply rod
            glColor3f(1,1,1);
            glLineWidth(3);
            glBegin(GL_LINES);
                glVertex2f(cartpos,0);
                glVertex2f(cartpos+RODLENGTH*sin(rodangle),RODLENGTH*cos(rodangle));
            glEnd();
            
        //Display V vs action plot
    	glViewport(winw/2,3*winh/4.0,winw/2,winh/4.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(-1.5,1.5,-1*viewscales,.4*viewscales);
        
        glColor3f(.3,.3,.3);
        glLineWidth(1);
        glBegin(GL_LINES);
            for(j=-10;j<0;j+=.2){
                glVertex2f(-2,j);
                glVertex2f(2,j);
            }
            glVertex2f(0,1);
            glVertex2f(0,-10);
            glColor3f(1,1,1);
            glVertex2f(-10,0);
            glVertex2f(10,0);
        glEnd();
        
        glColor3f(1,1,0);
        glLineWidth(3);
        glBegin(GL_LINES);
            for(i=0;i<ACTIONNUM;i++){
                glVertex2f(actionmap[i]-0.5/ACTIONNUM,Vaction[i]);
                glVertex2f(actionmap[i]+0.5/ACTIONNUM,Vaction[i]);
            }
        glEnd();
        
        glLineWidth(6);
        glColor3f(.2,.2,1);
        glBegin(GL_LINE_STRIP);
            glVertex2f(-1.5,10); glVertex2f(-1.5,-10);
        glEnd();
            
        //Display V vs time plots
    	glViewport(0,2*winh/4.0,winw,winh/4.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0,PLOTWIDTH,-2*viewscales,.4*viewscales);
        
        glColor3f(.3,.3,.3);
        glLineWidth(1);
        glBegin(GL_LINES);
            for(i=-20;i<0;i++){
                glVertex2f(0,i);
                glVertex2f(PLOTWIDTH,i);
            }
            glColor3f(1,1,1);
            glVertex2f(0,0);
            glVertex2f(PLOTWIDTH,0);
        glEnd();
        
        glPointSize(2);
        glLineWidth(3);
        glBegin(GL_LINE_STRIP);
            glColor3f(0,1,0);
            for(i=0;i<historyindex;i++)glVertex2f(i,history[0][i]);
        glEnd();
        glBegin(GL_LINE_STRIP);
            glColor3f(1,0,0);
            for(i=0;i<historyindex;i++)glVertex2f(i,history[1][i]);
        glEnd();
        
        glLineWidth(6);
        glColor3f(.2,.2,1);
        glBegin(GL_LINES);
            glVertex2f(0,.4*viewscales); glVertex2f(PLOTWIDTH,.4*viewscales);
        glEnd();
        
        //Display action/rodangle vs time plots
        
    	glViewport(0,winh/4.0,winw,winh/4.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0,PLOTWIDTH,-1.1,1.1);
        
        glColor3f(1,1,1);
        glLineWidth(1);
        glBegin(GL_LINES);
            glVertex2f(0,0);
            glVertex2f(PLOTWIDTH,0);
        glEnd();
        
        glLineWidth(3);
        glBegin(GL_LINE_STRIP);
            glColor3f(1,1,0);
            for(i=0;i<historyindex;i++)glVertex2f(i,history[2][i]);
        glEnd();
        glBegin(GL_LINE_STRIP);
            glColor3f(0,1,0);
            for(i=0;i<historyindex;i++)glVertex2f(i,history[3][i]);
        glEnd();
        
        glLineWidth(6);
        glColor3f(.2,.2,1);
        glBegin(GL_LINE_STRIP);
            glVertex2f(0,1.1); glVertex2f(PLOTWIDTH,1.1);
        glEnd();
        
        //Display Ravg vs time plots
    	glViewport(0,0,winw,winh/4.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0,resetcount,0,TWOPI/6);
        glColor3f(.3,.3,.3);
        glLineWidth(1);
        glBegin(GL_LINES);
            for(i=100;i<simtime;i+=100){
                glVertex2f(i,0);
                glVertex2f(i,TWOPI/6);
            }
        glEnd();
        
        glLineWidth(3);
        glBegin(GL_LINE_STRIP);
            glColor3f(.6,.6,1);
            for(i=0;i<resetcount;i++)glVertex2f(i,Ravg[i]);
        glEnd();
        
        glLineWidth(6);
        glColor3f(.2,.2,1);
        glBegin(GL_LINES);
            glVertex2f(0,TWOPI/6); glVertex2f(resetcount,TWOPI/6);
        glEnd();
        
        glutSwapBuffers();
    }
    
    if(!pausebit){
        applyaction();
        memindex=(memindex+1)%MEMSIZE;
        stateID(stateMem[memindex]);
        chooseaction();
        
        history[1][historyindex]=Vaction[actionMem[(memindex+MEMSIZE-1)%MEMSIZE]]; //Approximated V
        vactioncalc((memindex+MEMSIZE-1)%MEMSIZE,RLTYPE,Vaction); //Target V
        history[0][historyindex]=Vaction[actionMem[(memindex+MEMSIZE-1)%MEMSIZE]]; //Target V
        history[2][historyindex]=-cartpos/RODLENGTH;
        history[3][historyindex]=rodangle/(TWOPI/4);
        
        VdiffMem[memindex-1]=abs(history[1][historyindex]-history[0][historyindex]);
        historyindex++;
        
        if(historyindex==PLOTWIDTH)historyindex=0;
        if(MEMSIZE<=2)updateV(1-memindex,RLTYPE);
        else{
            int maxindex=min(simtime,MEMSIZE-1);
            for(i=0;i<min(REPLAYNUM,maxindex+2);i++){
                int replayi=rand()%(maxindex+1);
                while(VdiffMem[replayi]*randuniform(0,1)*2<REPLAYSKIPTHRESH){
                    replayi=rand()%(maxindex+1);
                }
                updateV(replayi,RLTYPE);
            }
        }
        
        if(abs(cartpos)>RODLENGTH || abs(rodangle)>TWOPI/4.0){
            if(resetcount==RESETCOUNTLIMIT-1){
                cout<<"\tSimulation timeout: Forcing reset.\n";
                resetall();
            }
            else resetproblem();
            stateID(stateMem[memindex]);
            chooseaction();
        }
        simtime++;
        if(simtime<EPSILONDECAYTIME)epsilon-=(EPSILON1-EPSILON2)/EPSILONDECAYTIME;
    }
}

void resize(int x,int y){
    if(x>0)winw=x;
    if(y>0)winh=y;
	glViewport(0,0,winw,winh);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-viewradius*winw/winh,viewradius*winw/winh,viewradius,-viewradius);
}

void init() 
{
    srand(time(NULL));
	glClearColor(0.0, 0.0, 0.0, 1.0);
    logtimestamp=time(NULL);
    cout<<"\nSimulation timestamp = "<<logtimestamp;
    
    //Initialize room
    int i;
    
    /*for(i=0;i<ROOMVERTICES;i++){
        float r=2*(rand()%1000)/1000.0+1;
        room[i][0]=r*cos(TWOPI*i/(float)ROOMVERTICES);
        room[i][1]=r*sin(TWOPI*i/(float)ROOMVERTICES);
    }
    */
    
    //Initialization
    nnwidth[0]=INPUTNUM; //Input layer
    for(i=1;i<NNDEPTH;i++)nnwidth[i]=NNWIDTH; //Hidden layers
    nnwidth[NNDEPTH]=ACTIONNUM; //Output layer
    
    resetcount=0;
    episodelength=1;
    resetall();
    stateID(stateMem[memindex]);
    chooseaction();
    
    printf("\n%06i",resetcount);
}
/*
void mouseclick(int button, int clickstate, int x, int y){   
    if(clickstate==GLUT_DOWN && abs(viewradius*((float)x*2.0/winw-1.0)-pcharger[0])<BOTSIZE/2.0 && abs(viewradius*((float)y*2.0/winh-1.0)-pcharger[1])<BOTSIZE/2.0)chargergrabbed=true;
    else if(clickstate==GLUT_UP)chargergrabbed=false;
}

void mousemotion(int x,int y){
    if(chargergrabbed){
        pcharger[0]=viewradius*(2.0*(float)x/winw-1);
        pcharger[1]=viewradius*(2.0*(float)y/winh-1);
    }
}
*/

void keyfunc(unsigned char key,int x,int y){
    switch(key){
        case ' ': dispbit=!dispbit; break;
        case 'p': pausebit=!pausebit; break;
        case '=': viewscales/=1.2; break;
        case '-': viewscales*=1.2; break;
        case ',': rodangle-=.5; break;
        case '.': rodangle+=.5; break;
        case 'r':
            cout<<"\tManual reset\n";
            resetall();
            break;
        case 'd': PrintNN(nnwidth,NNDEPTH,weight,bias);
        case 's': stuck=true;
    }
}

int main(int argc, char **argv) 
{
	glutInit(&argc, argv);                                                      // GLUT initialization
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);                                // Display Mode
	glutInitWindowSize(winw,winh);					                            // set window size
	glutCreateWindow("Reinforcement Learning Simulation");							// create Window
	glutDisplayFunc(disp);									                    // register Display Function
	glutIdleFunc(disp);
	glutReshapeFunc(resize);
	glutKeyboardFunc(keyfunc);
//	glutMouseFunc(mouseclick);
//	glutMotionFunc(mousemotion);
	//glutSpecialFunc(specialkeyfunc);
	//glutKeyUpFunc(keyupfunc);
	init();
	glutMainLoop();												                // run GLUT mainloop
	return 0;
}
