#author: Marcel H. Schubert
#
#content: preprocess data into a reduced format for input as features

library(magrittr)
library(lubridate)
library(tidyverse)
### Use legacy nest and unnest
nest <- nest_legacy
unnest <- unnest_legacy


#function for rowwise na filtering
remove_na_rowise = function(df){
  
  df[apply(df, 1, function(x) any(is.na(x))),]
}

##non-violent
load("../../data/preprocessed/Table_construction.Rdata")

propub <- read.csv("../../data/raw_data/pro_publica/compas-scores.csv")


features_filt = features %>%
  inner_join(
    data_before %>% 
      select(person_id, screening_date, people) %>%
      unnest() %>%
      select(person_id, screening_date, race, sex, name),
    by = c("person_id","screening_date")
  ) %>%
  inner_join(features_on, by = c("person_id","screening_date")) %>%
  inner_join(outcomes, by = c("person_id","screening_date")) %>%
  mutate(
    race_black = if_else(race=="African-American",1,0),
    race_white = if_else(race=="Caucasian",1,0),
    race_hispanic = if_else(race=="Hispanic",1,0),
    race_asian = if_else(race=="Asian",1,0),
    race_native = if_else(race=="Native American",1,0), # race == "Other" is the baseline
    
    #baseline is single
    is_married = ifelse(marital_status == 'Married', 1, 0),
    is_divorced = ifelse(marital_status == 'Divorced', 1, 0),
    is_widowed = ifelse(marital_status == 'Widowed', 1, 0),
    is_separated = ifelse(marital_status == 'Separated', 1, 0),
    is_sig_other = ifelse(marital_status == "Significant Other", 1, 0),
    is_marit_unknown = ifelse(marital_status == "Unknown", 1, 0),
    
    
    
    p_jail30 = pmin(p_jail30,5),
    p_prison30 = pmin(p_jail30,5),
    p_prison = pmin(p_prison,5),
    p_probation = pmin(p_probation,5),
    
    p_age_first_offense,
    p_juv_fel_count = pmin(p_juv_fel_count,2),
    p_felprop_violarrest = pmin(p_felprop_violarrest,5),
    p_murder_arrest = pmin(p_murder_arrest,3),
    p_felassault_arrest = pmin(p_felassault_arrest,3),
    p_misdemassault_arrest = pmin(p_misdemassault_arrest,3),
    #p_famviol_arrest = pmin(p_famviol_arrest,3),
    p_sex_arrest = pmin(p_sex_arrest,3),
    p_weapons_arrest = pmin(p_weapons_arrest,3),
    p_n_on_probation = pmin(p_n_on_probation,5),
    p_current_on_probation = pmin(p_current_on_probation,5),
    p_prob_revoke = pmin(p_prob_revoke,5),
    
    # Subscales:
    crim_inv_arrest = p_arrest+ 
      p_jail30+
      p_prison+
      p_probation,
    
    crim_inv_charge = p_charge+ 
      p_jail30+
      p_prison+
      p_probation,
    
    # Subscales:
    vio_hist = p_juv_fel_count+
      p_felprop_violarrest+
      p_murder_arrest+
      p_felassault_arrest+
      p_misdemassault_arrest+
      #p_famviol_arrest+ #because no observations have nonzero for this
      p_sex_arrest+
      p_weapons_arrest,
    history_noncomp = p_prob_revoke+
      p_probation+p_current_on_probation+
      p_n_on_probation,
    
    
    # Filters (TRUE for obserations to keep)
    filt1 = `Risk of Recidivism_decile_score` != -1 & `Risk of Violence_decile_score` != -1, 
    filt2 = offenses_within_30 == 1,
    filt3 = !is.na(current_offense_date), 
    filt4 = ifelse(filt3, current_offense_date <= current_offense_date_limit, screening_date < current_offense_date_limit) , 
    filt5 = p_current_age > 18 & p_current_age <= 65, 
    filt6a = crim_inv_arrest == 0,
    filt6b = crim_inv_charge == 0,
    
  )

propub <- propub[, 1:46]

propub <- propub %>% rename(
  person_id = id,
  screening_date = compas_screening_date
  
)
propub$screening_date <- as_date(ymd_hms(as.character(propub$screening_date), truncated = 3))

features_filt <- left_join(features_filt,
                           propub[, c('person_id', "screening_date", "is_recid", "is_violent_recid")],
                           by = c('person_id', 'screening_date'))

features_filt <- features_filt %>% rename(
  recid_proPub = is_recid,
  recid_violent_proPub = is_violent_recid
) 
features_filt <- features_filt %>% mutate(
    filt_pp_recid = !is.na(recid_proPub),
    filt_pp_violent_recid = !is.na(recid_violent_proPub),
    uid = paste(person_id, screening_date, sep='_'),
    current_offense_date = as_date(ifelse(is.na(current_offense_date), screening_date, current_offense_date))
  )

#remove all data with missing compas values
features_filt = features_filt[features_filt$filt1,]

#remove all data for which we have an offense date accompanying the screening date or the other way round 
features_no_two = features_filt[features_filt$filt3,]


#make subset with those data points for which we have no 2-year period of observation data
features_red = features_filt[features_filt$filt3 & features_filt$filt4,]

#make subsets with datapoints for which we can use the proPublica recid-values
features_filt_pro_base = features_filt[features_filt$filt_pp_recid,]
features_filt_pro_viol = features_filt[features_filt$filt_pp_violent_recid,] 

features_no_two_pro_base = features_no_two[features_no_two$filt_pp_recid,]
features_no_two_pro_viol = features_no_two[features_no_two$filt_pp_violent_recid,]

features_red_pro_base = features_red[features_red$filt_pp_recid,]
features_red_pro_viol = features_red[features_red$filt_pp_violent_recid,]

##drop famviol as no non-zero observations
#features_filt = features_filt[, -c(14)]



#select columns of interest
names1 <- c("person_id","screening_date","before_cutoff_date","marital_status","race" ,"name" ,
  "filt1","filt2","filt3", "filt4" ,"filt5","filt6a" ,"filt6b", "filt_pp_recid","filt_pp_violent_recid")


features_filt = features_filt[, -c(which(colnames(features_filt) %in% names1))]
features_no_two = features_no_two[, -c(which(colnames(features_no_two) %in% names1))]
features_red = features_red[, -c(which(colnames(features_red) %in% names1))]

features_filt_pro_base = features_filt_pro_base[, -c(which(colnames(features_filt_pro_base) %in% names1))]
features_filt_pro_viol = features_filt_pro_viol[, -c(which(colnames(features_filt_pro_viol) %in% names1))]

features_no_two_pro_base = features_no_two_pro_base[, -c(which(colnames(features_no_two_pro_base) %in% names1))]
features_no_two_pro_viol = features_no_two_pro_viol[, -c(which(colnames(features_no_two_pro_viol) %in% names1))]

features_red_pro_base = features_red_pro_base[, -c(which(colnames(features_red_pro_base) %in% names1))]
features_red_pro_viol = features_red_pro_viol[, -c(which(colnames(features_red_pro_viol) %in% names1))]




##equal missing values as Trues sum to 0
print(sum(!(features_filt_pro_base %in% features_filt_pro_viol)))

ycolnames <- c("Risk of Failure to Appear_decile_score","Risk of Failure to Appear_raw_score",
               "Risk of Failure to Appear_score_text" ,"Risk of Recidivism_decile_score", "Risk of Recidivism_raw_score",
               "Risk of Recidivism_score_text" ,"Risk of Violence_decile_score", "Risk of Violence_raw_score",
               "Risk of Violence_score_text"  ,"recid", "recid_violent" , "recid_proPub" ,"recid_violent_proPub")

##nonsubsetted
write.csv(features_filt[, -c(which(colnames(features_filt) %in% ycolnames))],
          file = '../../data/preprocessed/input_features.csv')
write.csv(features_filt[, which(colnames(features_filt) %in% c(ycolnames, 'uid'))],
           file = '../../data/preprocessed/y_features.csv')

#subsetted with observational data 
write.csv(features_no_two[, -c(which(colnames(features_no_two) %in% ycolnames))],
          file = '../../data/preprocessed/input_features_no_two.csv')
write.csv(features_no_two[, which(colnames(features_no_two) %in% c(ycolnames, 'uid'))],
          file = '../../data/preprocessed/y_features_no_two.csv')

#subsetted with missing date
write.csv(features_red[, -c(which(colnames(features_red) %in% ycolnames))],
          file = '../../data/preprocessed/input_features_reduced_size.csv')
write.csv(features_red[, which(colnames(features_red) %in% c(ycolnames, 'uid'))],
          file = '../../data/preprocessed/y_features_reduced_size.csv')

#for nonsubsetted with proPublica recid
write.csv(features_filt_pro_base[,-c(which(colnames(features_filt_pro_base) %in% ycolnames))],
          file = '../../data/preprocessed/input_features_pP_recid.csv')
write.csv(features_filt_pro_base[, which(colnames(features_filt_pro_base) %in% c(ycolnames, 'uid'))],
          file = '../../data/preprocessed/y_features_pP_recid.csv')

#for no_two with proPublica recid
write.csv(features_no_two_pro_base[, -c(which(colnames(features_no_two_pro_base) %in% ycolnames))],
          file = '../../data/preprocessed/input_features_pP_recid_no_two.csv')
write.csv(features_no_two_pro_base[, which(colnames(features_no_two_pro_base) %in% c(ycolnames, 'uid'))],
          file = '../../data/preprocessed/y_features_pP_recid_no_two.csv')

#for subsetted with proPublica recid
write.csv(features_red_pro_base[, -c(which(colnames(features_red_pro_base) %in% ycolnames))],
          file = '../../data/preprocessed/input_features_pP_recid_reduced_size.csv')
write.csv(features_red_pro_base[, which(colnames(features_red_pro_base) %in% c(ycolnames, 'uid'))],
          file = '../../data/preprocessed/y_features_pP_recid_reduced_size.csv')


save(data_before, data_on, data_after, data_before_on,
     features, features_before_on, features_on, outcomes,  
     compas_df_wide,
     current_offense_date_limit,
     features_filt,
     features_no_two,
     features_red,
     features_filt_pro_base,
     features_filt_pro_viol,
     features_no_two_pro_base,
     features_no_two_pro_viol,
     features_red_pro_base,
     features_red_pro_viol,
     file = "../../data/preprocessed/feature_data.Rdata")
