/* file: fdints.i */

/* name of module to use*/
%module fdints

%{
#include <stdlib.h>
#include "fdints.h"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_fermi_dirac.h>
%}

%include "fdints.h"
//%include <gsl/gsl_sf_bessel.h>
//%include <gsl/gsl_sf_fermi_dirac.h>

