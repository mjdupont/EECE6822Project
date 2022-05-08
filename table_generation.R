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

output_df <- df[-c(1, 2, 5)]
output_df <- output_df %>% mutate(Station=replace(Station, Station=='Natural Energy Laboratory of Hawaii Authority', 'NEL Hawaii'))

output_df <- output_df %>% rename(Persistence_Perf = Persistence_Performance) %>% rename(Model_Perf = Model_Performance)

output_df_type1 <- output_df[output_df$Station %in% c('Titusville FL',
                                                       'NEL Hawaii',
                                                       'Millbrook NY'),]
tab_df(
  output_df_type1,
  title="Appendix A. Model Performance Results - Non-Irradiance Only Predictors"
)

output_df_type2 <- output_df[output_df$Station %in% c('Table Mountain Boulder CO',
                                                      'Goodwin Creek MS',
                                                      'Bondville IL'
                                                      ),]
tab_df(
  output_df_type2,
  title="Appendix B. Model Performance Results - Non-Irradiance+Irradiance Predictors"
)

output_df_type3 <- output_df[output_df$Station %in% c('Sterling Virginia',
                                                      'Seattle Washington',
                                                      'Salt Lake City Utah', 
                                                      'Hanford California'
),]
tab_df(
  output_df_type3,
  title="Appendix C. Model Performance Results - Irradiance Only Predictors"
)


tab_df(
  output_df,
  title="Table 2. Model Performance Results"
)

df_melted <- read_csv("results_melted.csv")
tab_df(
  df_melted,
  title="Table 2. Model Performance Results"
)
