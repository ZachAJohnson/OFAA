#include "fdints.h"
#include <stdio.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <cmath>
#include <math.h>


 double fthreehalf ( double eta)
{
  double y;
  if (eta<-30) { //Accurate to 1e-14
    y = exp(eta);
  }
  else if (eta>1e8){ //Accurate to 1e-14
    y = 8*pow(eta,2.5)/(15*sqrt(M_PI));
  }
  else {
    y = gsl_sf_fermi_dirac_3half(eta);  
  }
  
  return y;
}

 double fonehalf ( double eta)
{
  /* This is somehow broken! Does not agree with analytic limits from
  https://arxiv.org/pdf/0811.0116.pdf

  double y = gsl_sf_fermi_dirac_mhalf(eta);

  Instead, use I3/2, which is right!
  */

  double y;
  if (eta<-30) {
    y = exp(eta);
  }
  else if (eta>1e8) { 
    y = 4*pow(eta,1.5)/(3*sqrt(M_PI));
  }
  else {
    double epsilon = 1e-6;
    if (eta> 1e-6) {
     epsilon = 1e-6*eta;
    };
    y= (fthreehalf(eta+epsilon)-fthreehalf(eta-epsilon))/(2*epsilon);
  }

  return y;
}

double fminusonehalf ( double eta)
{
  double y;
  if (eta<-30) { 
    y = exp(eta);
  }
  else if (eta>1e8){
    y = 2*sqrt(eta)/sqrt(M_PI);
  }
  else {
    y = gsl_sf_fermi_dirac_mhalf(eta);
  }

  return y;
}
