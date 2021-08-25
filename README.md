# Content
```bash
├───code
│   └───preprocess_data (R code for preprocessing of data)
└───data
    ├───preprocessed (preprocessed features and y in csv format)
    └───Raw_data
        ├───compas-analysis (raw ProPublica COMPAS DB and extracted tables as csv)
        ├───probation (probation data from Rudin et al. 2019 )
        └───pro_publica (ProPublica processed data sets)
```

## Preprocessed Data

The folder holds .Rdat files which contain the data processed by Table_construction.Rmd. The data is then further processed to contain only the features of relevance. The features are then saved in csv-files which may also be found in this folder. In short, there are two distinct files:
input_features: These are features as compiled by the code in _Table_construction.Rmd_ and _functions.R_. These features are the ones most likely used by COMPAS. In principle, it is possible to concsatruct additional features, such as the frequency of offenses etc. Thas, however requires some additional thought beforehand.
y_features: There are multiple possible y-values one can predict but only one y-feature should be used at a time

### Preprocessed .csv-Files
The input_feature-files hold all the input features while the y_features-files hold the output to train against.
Of those two files there are different versions which all hold the same amount of features but contain a different number of rows, i.e. different amounts of training data.
1. features.csv: contains all data points, including those for which we have a COMPAS _screening_date_ but no _current_offense_date_. If there is a _NA_ for _current_offense_date_, _screening_date_ was used to fill it.
2. features_no_two_year.csv contains those datapoints for which we have a current offense date but not necessarily two years of observational information after the screening process.
3. features_reduced_size.csv: This is the __preferred__ feature set. It reflects that of the literature, where we excluded all those observations for which _current_offense_date_ is _NA_
4. \*\_pP_recid: ProPublica calculates their y-variable for recidivism slightly different than Rudin et al. 2019 (and by extension we) do. However, as they do not provide their code we cannot search for the underlying reason. The _y\_\*\_pP_recid_-files expose their recidivism-variable instead of ours. The full _features\_\*\_pP_recid.csv_ has fewer rows than before as does the reduced data set.

The files have the following dimensions:
input_features.csv & y_features.csv: \[12366, 37\] & \[12366, 11\]
input_features_pP_recid.csv & y_features_pP_recid.csv: \[11742, 37\] & \[11742, 11\]
input_features_no_two.csv & y_features_no_two.csv: \[9042, 37\] & \[9042, 11\]
input_features_no_two_pP_recid.csv & y_features_no_two_pP_recid.csv: \[8627, 37\] & \[8627, 11\]
input_features_reduced_size.csv & y_features_reduced_size.csv: \[5759, 37\] & \[5759, 11\]
input_features_pP_recid_reduced_size.csv & y_features_pP_recid_reduced_size.csv: \[5553, 37\] & \[5553, 11\]


### Features in the feature-files:

Feature Name | Type | Values | Explanation
-------------|------|--------|-------------
uid | string | - | Unique identifier; Concat of id and screening date
first_offense_date | string | - | Date of first offense commited
current_offense_date | string - | Date of the current offense in question for which COMPAS screening took place
offenses_within_30 | integer | count | Count all offenses that occured up unitl 30 days prior to screening date 
p_felony_count_person | integer | count | Prior number of felonies committed by person
p_misdem_count_person | integer | count | Prior number of misdemeanours committed by person
p_charge_violent| integer | count | Number of charges against individual falling under violent crimes/offenses   
p_current_age| integer | age (not normalized) | Age in years of the individual when committing the offense
p_age_first_offense| integer | age (not normalized) | Age when committing the first offense (static)
is_married | integer | bool | baseline is 'single'
is_divorced | integer | bool | sbaseline is 'single'
is_widowed | integer | bool | sbaseline is 'single'
is_separated | integer | bool | sbaseline is 'single'
is_sig_other| integer | bool | sbaseline is 'single'
is_marit_unknown | integer | bool | sbaseline is 'single'
__History of Violence Subscale Items__ | | | Items are calculatated with prior charges (we can also calcualte them with arrests instead (less clean but also fewer zeros))
p_juv_fel_count| integer | count | Prior number of felonies committed by person while the individual was still juvenile | integer | count | Prior number of felonies committed by person while the individual 
p_felprop_violarrest | integer | count | Prior violent felony property offense arrests
p_murder_arrest | integer | count | Prior voluntary manslaughter/murder arrests
p_felassault_arrest | integer | count | Prior felony assault offense arrests (excluding murder, sex, or domestic violence)
p_misdemassault_arrest | integer | count | Prior misdemeanor assault offense arrests (excluding murder, sex, domestic violence)
p_famviol_arrest | integer | count | Prior family violence arrests
p_sex_arrest  | integer | count | Prior misdemeanor assault offense arrests (excluding murder, sex, domestic violence)
p_famviol_arrest | integer | count | Prior family violence arrests        
p_weapons_arrest | integer | count | Prior weapons offense arrest 
__History of Noncompliance Subscale Items__ | | | 
p_n_on_probation | integer | count | Prior number of offenses while on probation 
p_current_on_probation | Boolean | [0,1] | Current offense committed while on probation
p_prob_revoke | integer | count | Number of times probation terms were violeted or probation was revoked
__Criminal Involvment Subscale Items__ | | | For this subscale, either the value p_charge XOR p_arrest is used
p_charge | integer | count | Prior number of charges
p_arrest | integer | count | Prior number of arrests               
p_jail30 | integer | count | Prior number of times sentenced to jail 30 days or more
p_prison30 | integer | count | Prior number of times sentenced to prison 30 days or more
p_prison | integer | count | Prior number of times sentenced to prison
p_probation | integer | count | Prior number of times sentenced to propbation as an adult
| | |
__Others__ | | 
sex | string | "Female", "Male" | Gender
is_misdem | integer | [0,1] | If all charges connected to the current offenses are only misdemeanures = 1, otherwise 0 (i.e. at least one charge is in regards to a felony)
race_black | integer | [0,1] | Individual is black = 1 (baseline is race_other)
race_white | integer | [0,1] | Individual is white = 1 (baseline is race_other)            
race_hispanic | integer | [0,1] | Individual is hispanic = 1 (baseline is race_other)
race_asian | integer | [0,1] | Individual is asian = 1 (baseline is race_other)
race_native | integer | [0,1] | Individual is native = 1 (baseline is race_other)           
crim_inv_arrest| integer | count | Criminal Involvment Scale calculated from features as outlined above. Scale is a simple sum of count-based-features. Uses p_charge
crim_inv_charge| integer | count | Criminal Involvment Scale calculated from features as outlined above. Scale is a simple sum of count-based features. Uses p_arrest
vio_hist | integer | count | History of Violence Subscale calculated from features as outlined above. Scale is simple sum of count-based features
history_noncomp | integer | count | History of Noncompliance Subscale calculated from features as outlined above. Scale is simple sum of count-based features

### y-Features

Feature Name | Type | Values | Explanation
-------------|------|--------|-------------
Risk of Failure to Appear_score_text | string | low, medium, high |  Formed from decile score. Medium and high necessitate special coonsideration for incaracaration decision
Risk of Failure to Appear_decile_score | integer | 1-10 | Normed raw score. Normed by underlying data we do not have but could approximate. The normation is done within the county and within age and gender as well as race groups.
Risk of Failure to Appear_raw_score | integer | 11-48 | COMPAS score Failure to appear
Risk of Recidivism_score_text | string | low, medium, high |  Formed from decile score. Medium and high necessitate special coonsideration for incaracaration decision
Risk of Recidivism_decile_score | integer | 1-10 | Normed raw score. Normed by underlying data we do not have but could approximate. The normation is done within the county and within age and gender as well as race groups.
Risk of Recidivism_raw_score | double | (-3) - (2.36) | COMPAS score risk of Recidivism (feature of most interest)
Risk of Violence_score_text | string | low, medium, high |  Formed from decile score. Medium and high necessitate special coonsideration for incaracaration decision
Risk of Violence_decile_score | integer | 1-10 | Normed raw score. Normed by underlying data we do not have but could approximate. The normation is done within the county and within age and gender as well as race groups. 
Risk of Violence_raw_score | double | (-4.63) - (0.5) | COMPAS score risk of violence (feature of second highest interest)
recid | integer | [0,1] | Individual is recidivistic within two years after screening = 1
recid_violent | integer | [0,1] | Individual is violent recidivistic within two years after screening = 1
recid_proPub | integer | [0,1] | Individual is recidivistic within two years after screening = 1 as calculated by ProPublica
recid_violent_proPub | integer | [0,1] | Individual is violent recidivistic within two years after screening = 1 as calculated by ProPublica
