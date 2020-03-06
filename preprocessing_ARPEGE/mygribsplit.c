#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
/*
 * This programs splits a ARPEGENH grib2 file producing one
 * file for every grib2 fields, each field is also converted
 * to a ecmwf-compatible grib2 file that can be processed by cdo
 * author : ludovic AUGER ludovic.auger@meteo.Fr
 */

//#define BREAD(a,b,c) ((isection==a)&&(isecint==b-1) ? bitadd(fic1,fic2,&byteptr,c,0):-9999)
#define BREAD(d,a,b,c) if ((isection==a)&&(isecint==b))  {d=bitadd(fic1,fic2,&isecint,c,0);lread=1;}
#define BSKIP(a,b,c) if ((isection==a)&&(isecint==b))  {bskip(fic1,&isecint,c);lread=1;}
#define BSET(a,b,c) if ((isection==a)&&(isecint==b))  {bset(fic1,fic2,&isecint,c);lread=1;}
#define BREADSET(d,a,b,c) if ((isection==a)&&(isecint==b))  {d=bset(fic1,fic2,&isecint,c);lread=1;}


int bitadd(FILE *fic1,FILE *fic2,int *byteptr,int nbytes,int val);
void bskip(FILE *fic1,int *byteptr,int nbytes);
int bset(FILE *fic1,FILE *fic2,int *byteptr,int val);

void bitskip(FILE *fic1,FILE *fic2,int *byteptr,int nbytes);

void main(int argc, char *fic[]) {
	int byteptr,c1,endfile,secsize,secend,nsection,secdeb,indexsec,ndata,discipline,number,category;
	int c2,c3,c4,c5,c6,c7,c8,zi,zinew,ieof,lshift,i,nfield,h1,h2,h3,h4,htot,stype,shiftdata;
	int isection,nsecl,isecint,lread,gridtype;
	FILE *fic1,*fic2;
	char ficnam[80];
	char subbuff[16];
	char csection2[600];
	char tmpfilename[100];
	fic1 = fopen(fic[1], "r");
	ndata=1;
	lshift=1;
	ieof=0;
	nfield=0;
	while (ieof==0){
		nfield++;
		endfile=0;
		byteptr=0;
		secsize=16;
		secend=16;
		nsection=0;
		secdeb=1;
		shiftdata=0;

		c1 = fgetc(fic1);
		if (feof(fic1)) {
			ieof=1;
			break;
		}
		byteptr++;
		sprintf(tmpfilename, "%s_tmpfile",fic[2]);	
		fic2 = fopen(tmpfilename, "wb");
		fputc(c1,fic2);
		for (i=2;i<=8;i++){
			c1 = fgetc(fic1);
			fputc(c1,fic2);
			byteptr++;	
			if (byteptr==7) {
				//	printf("discipline %d\n",c1);
				discipline=c1;
			}
		}
		if (feof(fic1)) {
			ieof=1;
			break;
		}


		//message size read and write
		zi=bitadd(fic1,fic2,&byteptr,8,-24);

		// loop on sections
		for(isection=1;isection<=7;isection++){

			//reading first four bytes (section length)
			if (isection==3) {
				nsecl=bitadd(fic1,fic2,&byteptr,4,-24);
			}else{
				nsecl=bitadd(fic1,fic2,&byteptr,4,0);
			}
			isecint=5;

			//loop inside a given section
			while(isecint<=nsecl) {
				lread=0;
				BSET(3,13,0)
				BREADSET(gridtype,3,14,40)
				//printf("avant bskip %d\n",isecint);
				BSKIP(3,73,24)
				//printf("apres bskip %d\n",isecint);
				BREAD(number,4,11,1)
				BREAD(category,4,10,1)
				BREAD(htot,4,25,4)
				BREAD(stype,4,23,1)
				if (isection==2) {
                                c1 = fgetc(fic1);
			       	fputc(c1,fic2);
				csection2[isecint]=c1;
				isecint++;
                                lread=1;
				}
				//BREAD(gridtype,3,14,2)
				//case no specific reading is done, need to advance to next byte.
				if (lread==0) { c1 = fgetc(fic1); fputc(c1,fic2);isecint++;}
			}
		}
		memcpy( subbuff, &csection2[8], 15 );
		if (strcmp(subbuff,"H00000HUMI.SPEC")==0) {
			discipline=0;
			category=1;
			number=0;
			htot=0;
			stype=103;
		}
		if (strcmp(subbuff,"H00000HUMI_RELA")==0) {
			discipline=0;
			category=1;
			number=1;
			htot=0;
			stype=103;
		}
		//reading last four grib2 bytes
		for(isecint=1;isecint<=4;isecint++) {
			c1 = fgetc(fic1);
			byteptr++;	
			fputc(c1,fic2);
		}


		if (htot>20000000) htot=0;
		if (htot>10000) htot=htot/10000;
		if (gridtype==53) {
			sprintf(ficnam,"mv -f %s %s.t%d.l%3.3d.grb.%d.%d.%d.spectral",tmpfilename,fic[2],stype,htot,discipline,category,number);
		}else{
			sprintf(ficnam,"mv -f %s %s.t%d.l%3.3d.grb.%d.%d.%d.gp",tmpfilename,fic[2],stype,htot,discipline,category,number);
		}
		printf("processing grib2 field number %d with discipline.parameterCategory.parameterNumber : %d.%d.%d\n",nfield,discipline,category,number);
		fclose(fic2);
		system(ficnam);
	}

}

void bitskip(FILE *fic1,FILE *fic2,int *byteptr,int nbytes){
	int i,c,z;
	for (i=1;i<=nbytes;i++) {
		c = fgetc(fic1); 
	}
}
int bset(FILE *fic1,FILE *fic2,int *byteptr,int val){
	int i,c;
	c = fgetc(fic1); 
	fputc(val,fic2);
	(*byteptr)++;
        return c;
}
void bskip(FILE *fic1,int *byteptr,int nbytes){
	int i,c;
	for (i=1;i<=nbytes;i++) {
		c = fgetc(fic1); 
		(*byteptr)++;
	}
}

int bitadd(FILE *fic1,FILE *fic2,int *byteptr,int nbytes,int val){
	int i,c;
	unsigned long long int z,zr,za;
	unsigned long long int zc[8];
	z=0;
	zr=0;
	zc[0]=1;
	for (i=1;i<8;i++) {
		zc[i]=256*zc[i-1];
	}
	for (i=1;i<=nbytes;i++) {
		c = fgetc(fic1); 
		(*byteptr)++;
		zr=256*zr+c;
	}
	z=zr+val;
	for (i=1;i<=nbytes;i++) {
		za=z/zc[nbytes-i];
		fputc(za,fic2);
		z=z-za*zc[nbytes-i];
	}
	return zr;
}
