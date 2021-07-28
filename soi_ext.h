/************************************************************************
* soi.h is the include file for the esarp SAR processor.		*
************************************************************************/
/************************************************************************
* Creator: Evelyn J. Price	(Scripps Institution of Oceanography)	*
* Date   : 11/18/96							*
************************************************************************/
/************************************************************************
* Modification History							*
*									*
* Date									*
*									*
*  4/23/97- 	added parameters for orbit calculations: x_target,      *
*		y_target,z_target,baseline,alpha,sc_identity,		*
*		ref_identity,SC_clock_start,SC_clock_stop,              *
*		clock_start,clock_stop   				*
*		-DTS							*
*									*
* 4/23/97-	added parameters: rec_start, rec_stop			*
*		-EJP							*
*									*
* 8/28/97-	added parameters baseline_start baseline_end		*
*		alpha_start alpha_end					*
*									*
* 9/12/97	added clipi2 function to clip to short int		*
*									*
* 4/26/06	added nrows, num_lines					*
* 7/27/21   added external definition for gmtsar-gpu                   *
************************************************************************/
#ifndef SOI_H	    
#define SOI_H	    
#include <stdio.h>
#include <string.h> 
#include <math.h>
#include <time.h>
#include <stdlib.h>
#define SOL 299792456.0
#define PI 3.1415926535897932
#define PI2 6.2831853071795864
#define I2MAX 32767.
#define I2SCALE 4.e6
#define TRUE 1
#define FALSE 0
#define RW 0666
#define MULT_FACT 1000.0 
#define sgn(A) ((A) >= 0.0 ? 1.0 : -1.0)
#define clipi2(A) ( ((A) > I2MAX) ? I2MAX : (((A) < -I2MAX) ? -I2MAX : A) )
#define clipc(A) ( ((A)>255) ? 255 : (A) )
#define cliph(A) ( ((A)>15) ? 15 : (A) )
#include "sfd_complex.h"

extern char *input_file;
extern char *led_file;
extern char *out_amp_file;
extern char *out_data_file;
extern char *deskew;
extern char *iqflip;
extern char *off_vid;
extern char *srm;
extern char *ref_file;
extern char *orbdir;
extern char *lookdir;

extern int debug_flag;
extern int bytes_per_line;
extern int good_bytes;
extern int first_line;
extern int num_patches;
extern int first_sample;
extern int num_valid_az;
extern int st_rng_bin;
extern int num_rng_bins;
extern int nextend;
extern int nlooks;
extern int xshift;
extern int yshift;
extern int fdc_ystrt;
extern int fdc_strt;

/*New parameters 4/23/97 -EJP */
extern int rec_start;
extern int rec_stop;
/* End new parameters 4/23/97 -EJP */ 

/* New parameters 4/23/97 -DTS */
extern int SC_identity;	/* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS  (6)-Envisat_SLC  (7)-TSX (8)-CSK (9)-RS2 (10)-S1A*/
extern int ref_identity;	/* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS  (6)-Envisat_SLC  (7)-TSX (8)-CSK (9)-RS2 (10)-S1A*/
extern double SC_clock_start;	/* YYDDD.DDDD */
extern double SC_clock_stop;	/* YYDDD.DDDD */
extern double icu_start;       /* onboard clock counter */
extern double clock_start;     /* DDD.DDDDDDDD  clock without year has more precision */
extern double clock_stop;      /* DDD.DDDDDDDD  clock without year has more precision */
/* End new parameters 4/23/97 -DTS */

extern double caltone;
extern double RE;		/* Local Earth radius */
extern double raa;             /* ellipsoid semi-major axis - added by RJM */
extern double rcc;             /* ellipsoid semi-minor axis - added by RJM */
extern double vel1;		/* Equivalent SC velocity */
extern double ht1;		/* (SC_radius - RE) center of frame*/
extern double ht0;		/* (SC_radius - RE) start of frame */
extern double htf;		/* (SC_radius - RE) end of frame */
extern double near_range;
extern double far_range;
extern double prf1;
extern double xmi1;
extern double xmq1;
extern double az_res;
extern double fs;
extern double slope;
extern double pulsedur;
extern double lambda;
extern double rhww;
extern double pctbw;
extern double pctbwaz;
extern double fd1;
extern double fdd1;
extern double fddd1;
extern double sub_int_r;
extern double sub_int_a;
extern double stretch_r;
extern double stretch_a;
extern double a_stretch_r;
extern double a_stretch_a;

/* New parameters 8/28/97 -DTS */
extern double baseline_start;
extern double baseline_center;
extern double baseline_end;
extern double alpha_start;
extern double alpha_center;
extern double alpha_end;
/* End new parameters 8/28/97 -DTS */
extern double bparaa;               /* parallel baseline - added by RJM */
extern double bperpp;               /* perpendicular baseline - added by RJM */

/* New parameters 4/26/06 */
extern int nrows;
extern int num_lines;

/* New parameters 09/18/08 */
extern double TEC_start;
extern double TEC_end;
#endif /* SOI_H	*/
