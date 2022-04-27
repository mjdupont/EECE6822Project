################################################################################
# Matthew Dupont
# EECE 6822 Machine Learning
# Project 
################################################################################

suppressWarnings(RNGversion("3.5.3"))

################################################################################
############################## Load Libraries ################################## 
################################################################################

# Wrote this helper function to combine loading and installing libraries.
# "require" does the same as "library", loading the listed library. 
# However "require" returns false if the library isn't installed.
# This pattern tries to load a given library, and if the library can't be 
# loaded, installs it, then loads it.
libraryAndMaybeInstall <- function(packageName) {
  if(!require(packageName, character.only = TRUE)) {
    install.packages(packageName) 
    library(packageName, character.only = TRUE)
  }
}

##Libraries
libraryAndMaybeInstall('cluster')
libraryAndMaybeInstall('readxl')
libraryAndMaybeInstall('tidyverse')
libraryAndMaybeInstall('GGally')
libraryAndMaybeInstall('sjPlot')
libraryAndMaybeInstall('forecast')
libraryAndMaybeInstall('data.table')
libraryAndMaybeInstall('caret')
libraryAndMaybeInstall('randomForest')
libraryAndMaybeInstall('gains')
libraryAndMaybeInstall('pROC')
libraryAndMaybeInstall('ggpubr')

################################################################################
####################### Load and Clean/Prepare Data ############################
################################################################################

# *!* Note the below will only work for RStudio *!* 
# Source: https://stackoverflow.com/questions/13672720/r-command-for-setting-working-directory-to-source-file-location-in-rstudio

if (Sys.getenv("RSTUDIO") == "1")
{
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

df <- read_csv("best_results.csv")

output_df <- best_performance[-c(1, 2, 5)]
