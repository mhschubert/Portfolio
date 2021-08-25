compute_features = function(person_id,
                            screening_date,
                            first_offense_date,
                            current_offense_date,
                            offenses_within_30, 
                            before_cutoff_date,
                            arrest,
                            charge,
                            jail,
                            prison,
                            prob,
                            people,
                            violence = FALSE) {
  ### Computes features (e.g., number of priors) for each person_id/screening_date.

  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date) 
  
  out = list()
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  ### Other features
  
  # Number of felonies
  out$p_felony_count_person = ifelse(is.null(charge), 0, sum(charge$is_felony, na.rm = TRUE))

  # Number of misdemeanors
  out$p_misdem_count_person  = ifelse(is.null(charge), 0, sum(charge$is_misdem, na.rm = TRUE))

  # Number of violent charges
  out$p_charge_violent  = ifelse(is.null(charge), 0, sum(charge$is_violent, na.rm = TRUE))

  #p_current_age: Age at screening date
  out$p_current_age = floor(as.numeric(as.period(interval(people$dob,screening_date)), "years"))

  #p_age_first_offense: Age at first offense
  out$p_age_first_offense = floor(as.numeric(as.period(interval(people$dob,first_offense_date)), "years"))

  ### History of Violence

   # p_juv_fel_count
   out$p_juv_fel_count = ifelse(is.null(charge), 0, sum(charge$is_felony & charge$is_juv,na.rm=TRUE))

   # p_felprop_violarrest
   out$p_felprop_violarrest = ifelse(is.null(charge), 0,sum(charge$is_felprop_violarrest, na.rm = TRUE))

   #p_murder_arrest
   out$p_murder_arrest = ifelse(is.null(charge), 0, sum(charge$is_murder, na.rm = TRUE))

   #p_felassault_arrest
   out$p_felassault_arrest = ifelse(is.null(charge), 0, sum(charge$is_felassault_arrest, na.rm = TRUE))

   #p_misdemassault_arrest
   out$p_misdemassault_arrest = ifelse(is.null(charge), 0, sum(charge$is_misdemassault_arrest, na.rm = TRUE))

   #p_famviol_arrest
   out$p_famviol_arrest = ifelse(is.null(charge), 0, sum(charge$is_family_violence, na.rm = TRUE))

   #p_sex_arrest
   out$p_sex_arrest = ifelse(is.null(charge), 0, sum(charge$is_sex_offense, na.rm = TRUE))

   #p_weapons_arrest
   out$p_weapons_arrest =  ifelse(is.null(charge), 0, sum(charge$is_weapons, na.rm = TRUE))
  
   if(violence){
     # Attempted using arrests instead of charges to compute History of Violence
    if(is.null(charge)){
      out$p_juv_fel_count = 0
      out$p_felprop_violarrest = 0
      out$p_murder_arrest = 0
      out$p_felassault_arrest = 0
      out$p_misdemassault_arrest = 0
      out$p_famviol_arrest = 0
      out$p_sex_arrest = 0
      out$p_weapons_arrest = 0
    }else{ # we assume only 1 arrest/day. If arrest has at least one charge of the type we are concerned with
      # (e.g. felony) we consider it an arrest of that type
      # p_juv_fel_count
      juv_fel_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(juv_charges = sum(is_juv, na.rm=TRUE),
                  fel_charges = sum(is_felony), na.rm = TRUE)
  
      out$p_juv_fel_count = sum(juv_fel_arrests$juv_charges > 0 & juv_fel_arrests$fel_charges > 0,na.rm=TRUE)
  
      #p_felprop_violarrest
      felprop_violarrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(felprop_violcharges = sum(is_felprop_violarrest, na.rm=TRUE))
  
      out$p_felprop_violarrest = sum(felprop_violarrests$felprop_violcharges > 0, na.rm = TRUE)
  
      #p_murder_arrest
      murder_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(murder_charges = sum(is_murder, na.rm=TRUE))
  
      out$p_murder_arrest = sum(murder_arrests$murder_charges > 0, na.rm = TRUE)
  
      #p_felassault_arrest
      felassault_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(felassault_charges = sum(is_felassault_arrest, na.rm=TRUE))
  
      out$p_felassault_arrest = sum(felassault_arrests$felassault_charges > 0, na.rm = TRUE)
  
      #p_misdemassault_arrest
      misdemassault_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(misdemassault_charges = sum(is_misdemassault_arrest, na.rm=TRUE))
  
      out$p_misdemassault_arrest = sum(misdemassault_arrests$misdemassault_charges > 0, na.rm = TRUE)
  
      #p_famviol_arrest
      famviol_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(famviol_charges = sum(is_family_violence, na.rm=TRUE))
  
      out$p_famviol_arrest = sum(famviol_arrests$famviol_charges > 0, na.rm = TRUE)
  
      #p_sex_arrest
      sex_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(sex_charges = sum(is_sex_offense, na.rm=TRUE))
  
      out$p_sex_arrest = sum(sex_arrests$sex_charges > 0, na.rm = TRUE)
  
      #p_weapons_arrest
      weapons_arrests = charge %>%
        group_by(date_charge_filed) %>%
        summarize(weapons_charges = sum(is_sex_offense, na.rm=TRUE))
  
      out$p_weapons_arrest =  sum(weapons_arrests$weapons_charges > 0, na.rm = TRUE)
  
    }
   }
  
  
  ### History of Non-Compliance
  
  # Number of offenses while on probation
  out$p_n_on_probation = ifelse(is.null(charge) | is.null(prob), 0, count_on_probation(charge,prob))
  
  # Whether or not current offense was while on probation (two ways)
  if(is.null(prob)){
    out$p_current_on_probation = 0
  } else if(is.na(current_offense_date)) {
    out$p_current_on_probation = NA
  } else {
    out$p_current_on_probation = if_else(count_on_probation(data.frame(offense_date=current_offense_date),prob)>0,1,0)
  }
  
  # Number of times provation was violated or revoked
  out$p_prob_revoke =  ifelse(is.null(prob), 0, sum(prob$is_revoke==1 & prob$EventDate < current_offense_date))
  
  ### Criminal Involvement
  
  # Number of charges / arrests
  out$p_charge = ifelse(is.null(charge), 0, nrow(charge))
  out$p_arrest = ifelse(is.null(arrest), 0, length(unique(arrest$arrest_date)))
  
  # Number of times sentenced to jail/prison 30 days or more
  out$p_jail30 = ifelse(is.null(prison), 0, sum(jail$sentence_days >= 30, na.rm=TRUE))
  out$p_prison30 = ifelse(is.null(prison), 0, sum(prison$sentence_days >= 30, na.rm=TRUE))
  
  # Number of prison sentences
  out$p_prison =  ifelse(is.null(prison), 0, nrow(prison))
  
  # Number of times on probation
  out$p_probation =  ifelse(is.null(prob), 0, sum(prob$prob_event=="On" & prob$EventDate < current_offense_date, na.rm = TRUE))
  
  
  

  return(out)
}


compute_features_on = function(person_id,
                               screening_date,
                               first_offense_date,
                               current_offense_date,
                               offenses_within_30,
                               before_cutoff_date,
                               arrest,
                               charge,
                               jail,
                               prison,
                               prob,
                               people) {
  ### Computes features related to current offense
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date) 
  
  out = list()
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  out$is_misdem = ifelse(is.null(charge), NA, if_else(any(charge$is_misdem==1) & all(charge$is_felony==0),1,0))
  
  return(out)
}

custody_days_no_charge <- function(dat, current_offense_date){
  current_offense_date = as_date(current_offense_date)
  cust_days = ifelse(as_date(dat$in_custody[1])<current_offense_date,
         sum(dat$sentence_days)-as.numeric(as.period(interval(as_date(dat$in_custody[1]),
                                                               current_offense_date)),'days'),
         sum(dat$sentence_days))
  
  
  
  #print(dat)
  #print(paste('date current:', current_offense_date, sep=''))
  return(cust_days)
}

warning_test <- function(dat, date_next_offense){
  tmp <- dat %>% filter(as.Date(in_custody) < as.Date(date_next_offense)) %>%
    dplyr::arrange(in_custody) %>%
    summarize(custody_days = ifelse(length(in_custody) ==0, 0,##in case the jail/prison all belong to a later offense
                                    ifelse(as_date(out_custody[length(out_custody)])< as_date(date_next_offense), #test whether to count all
                                           sum(sentence_days), 
                                           ifelse(length(sentence_days)>1, #test whether there is more than one sentence
                                                  sum(sentence_days[1:(length(sentence_days)-1)]+ #if yes remove the last one
                                                        as.numeric(as.period(interval(in_custody,date_next_offense)),
                                                                   "days")),
                                                  as.numeric(as.period(interval(in_custody,date_next_offense)),
                                                             "days")
                                           )
                                    )
    ),
    sentence = length(in_custody)
    )  
  print('warning')
  print(dat)
  print(date_next_offense)
  print(tmp)
  
  
}

custody_info <- function(dat, date_next_offense){
  tmp <- dat %>% filter(as.Date(in_custody) < as.Date(date_next_offense)) %>%
    dplyr::arrange(in_custody) %>%
    summarize(custody_days = ifelse(length(in_custody) ==0, 0,##in case the jail/prison all belong to a later offense
                                    ifelse(as_date(out_custody[length(out_custody)])< as_date(date_next_offense), #test whether to count all
                                                sum(sentence_days), 
                                      ifelse(length(sentence_days)>1, #test whether there is more than one sentence
                                             sum(sentence_days[1:(length(sentence_days)-1)]+ #if yes remove the last one
                                                   as.numeric(as.period(interval(in_custody,date_next_offense)),
                                                                                                 "days")),
                                                       as.numeric(as.period(interval(in_custody,date_next_offense)),
                                                                  "days")
                                             )
                                         )
    ),
    sentence = length(in_custody)
    )  
  
  
  
  
  #print(tmp)
  #print(paste('date next:', date_next_offense, sep=''))
  return(tmp)
}

compute_outcomes = function(person_id,
                            screening_date,
                            first_offense_date,
                            current_offense_date,
                            before_cutoff_date,
                            arrest,
                            charge,
                            jail,
                            prison,
                            prob,
                            people){
  
  out = list()
  #print(person_id)
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(ifelse(is.na(current_offense_date), screening_date, current_offense_date))
  
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  if(!is.null(jail)){
    jail = jail %>% dplyr::arrange(in_custody)
    #add a day to account for the fact when release and imprison date are the same
    jail$sentence_days <- jail$sentence_days+1
    
  }
  if(!is.null(prison)){
    prison = prison %>% dplyr::arrange(in_custody)
  }
  
  if(is.null(charge)) {
    out$recid = 0
    out$recid_violent = 0
    out$recid_corr = 0
    out$recid_violent_corr = 0
    
    #add jail and prison information
    if(!is.null(jail)){

      out$custody_days_jail = custody_days_no_charge(jail, current_offense_date)
      out$custody_days_jail_viol = custody_days_no_charge(jail, current_offense_date)
      out$jail_sentences = dim(jail)[1]
      out$jail_sentences_viol = dim(jail)[1]
    }else{
      out$custody_days_jail = 0
      out$jail_sentences = 0
      out$custody_days_jail_viol = 0
      out$jail_sentences_viol = 0
    }
    if(!is.null(prison)){

      out$custody_days_prison = custody_days_no_charge(prison, current_offense_date)
      out$custody_days_prison_viol = custody_days_no_charge(prison, current_offense_date)
      out$prison_sentences = dim(prison)[1]
      out$prison_sentences_viol = dim(prison)[1]
    }else{
      out$custody_days_prison = 0
      out$prison_sentences = 0
      out$custody_days_prison_viol = 0
      out$prison_sentences_viol = 0
    }
    
  } else {
    
    # Sort charges in ascending order
    charge = charge %>% dplyr::arrange(offense_date)
    # General recidivism
    date_next_offense = charge$offense_date[1]
    
    ##jail
    if(!is.null(jail)){
      #print('jail')
      tmp = custody_info(jail, date_next_offense)

      out$custody_days_jail = tmp$custody_days
      out$jail_sentences = tmp$sentence
      
      
    }else{
      out$custody_days_jail = 0
      out$jail_sentences = 0
    }
    
    ##prison
    if(!is.null(prison)){
      #print('prison')
      tmp = custody_info(prison, date_next_offense)
      
      out$custody_days_prison = tmp$custody_days
      out$prison_sentences = tmp$sentence
    }else{
      out$custody_days_prison = 0
      out$prison_sentences = 0
    }
    
    
    prison_jail_days = out$custody_days_prison+out$custody_days_jail
    
    years_next_offense = as.numeric(as.period(interval(screening_date,date_next_offense)), "days")
    ##2 years are 2*365 days
    out$recid = if_else(years_next_offense <= 2*365, 1, 0)
    #take prison+jail time into account
    out$recid_corr = if_else(years_next_offense <= 2*365+prison_jail_days, 1, 0)
    
    
    
    
    # Violent recidivism
    date_next_offense_violent = filter(charge,is_violent==1)$offense_date[1]
    if(is.na(date_next_offense_violent)) {##no violent next offense
      out$recid_violent = 0
      out$recid_violent_corr = 0
      
      #add jail and prison information
      if(!is.null(jail)){

        out$custody_days_jail_viol = custody_days_no_charge(jail, current_offense_date)

        out$jail_sentences_viol = dim(jail)[1]
      }else{
        out$custody_days_jail_viol = 0
        out$jail_sentences_viol = 0
      }
      if(!is.null(prison)){

        out$custody_days_prison_viol = custody_days_no_charge(prison, current_offense_date)
        out$prison_sentences_viol = dim(prison)[1]
      }else{
        out$custody_days_prison_viol = 0
        out$prison_sentences_viol = 0
      }
      
    } else { ##violent next offense exists
      
      ##jail
      if(!is.null(jail)){
        tmp = custody_info(jail, date_next_offense_violent)
        
        out$custody_days_jail_viol = tmp$custody_days
        out$jail_sentences_viol = tmp$sentence
        
        
      }else{
        out$custody_days_jail_viol = 0
        out$jail_sentences_viol = 0
      }
      
      ##prison
      if(!is.null(prison)){
        tmp = custody_info(prison, date_next_offense_violent)
        
        out$custody_days_prison_viol = tmp$custody_days
        out$prison_sentences_viol = tmp$sentence
      }else{
        out$custody_days_prison_viol = 0
        out$prison_sentences_viol = 0
      }
      
      
      prison_jail_days = out$custody_days_prison_viol+out$custody_days_jail_viol
      
      years_next_offense_violent = as.numeric(as.period(interval(screening_date,date_next_offense_violent)), "days")
      out$recid_violent = ifelse(years_next_offense_violent <= 2*365, 1, 0)
      
      #take prison+jail time into account
      out$recid_violent_corr = ifelse(years_next_offense_violent <= 2*365+prison_jail_days, 1, 0)
    }
  }
  #print(as.data.frame(out))
  #flush.console()
  return(out)
}

count_on_probation = function(charge, prob){
  
  # Make sure prob is sorted in ascending order of EventDate
  
  u_charge = charge %>%
    group_by(offense_date) %>%
    summarize(count = n()) %>%
    mutate(rank = findInterval(as.numeric(offense_date), as.numeric(prob$EventDate)))  %>%
    group_by(rank) %>%
    mutate(
      event_before = ifelse(rank==0, NA, prob$prob_event[rank]),
      days_before = ifelse(rank==0, NA, floor(as.numeric(as.period(interval(prob$EventDate[rank],offense_date)), "days"))),
      event_after = ifelse(rank==nrow(prob), NA, prob$prob_event[rank+1]),
      days_after = ifelse(rank==nrow(prob),NA, floor(as.numeric(as.period(interval(offense_date, prob$EventDate[rank+1])), "days")))
    ) %>%
    mutate(is_on_probation = pmap(list(event_before, days_before, event_after, days_after), .f=classify_charge)) %>%
    unnest()
  
  return(sum(u_charge$count[u_charge$is_on_probation]))
}

