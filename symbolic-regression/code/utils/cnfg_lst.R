#@Author: Marcel H. Schubert
#this is a dictionary of dtw and partitional configurtions to test

require(dtwclust)



# Fuzzy preprocessing: calculate autocorrelation up to 2nd lag
acf_fun_multi <- function(series, ...) {
  lapply(series, function(x) {
    as.numeric(acf(x, lag.max = 9, plot = FALSE)$acf)
  })
}

##placeholder function so that no preprocessing is applied to fuzzy
##(needs to be adapted when using time series of differing lengths)
acf_fun_uni <- function(series,...){
  series
}

# Define overall configuration for multivariate
define_config_multi <- function(num_clust, window_size, typs = c("p")){
  cfgs <- compare_clusterings_configs(
    
    types = typs,  
    k = num_clust,
    
    controls = list(
      partitional = partitional_control(iter.max = 50L, nrep = 20L),
    ),
    
    preprocs = pdc_configs(type = "preproc",
                           none = list(),
                           # specify which should consider the shared ones
                           share.config = c("p")
    ),
    
    distances = pdc_configs(type = "distance",#tadpole is ignored
                            partitional = list(
                              dtw_basic = list(window.size = window_size, norm = c("L2")),
                              sdtw = list(window.size = window_size, norm = c("L2"),
                                          gamma = c(seq(0.001, 0.01, 0.0015), 0.1)),
                              gak = list(window.size = window_size, sigma = NULL)
                              )
    ),
    
    centroids = pdc_configs(type = "centroid",
                            partitional = list(pam = list(znorm = c(FALSE)),
                                               dba=list(norm = c('L2'), znorm = c(FALSE))
                                               )
                            )
    )
  
  print('done with define configs')
  return(cfgs)
}

# Define overall reduced configuration for multivariate
define_config_multi_red <- function(num_clust, window_size, typs = c("p")){
  cfgs <- compare_clusterings_configs(
    
    types = typs,  
    k = num_clust,
    
    controls = list(
      partitional = partitional_control(iter.max = 50L, nrep = 20L),
    ),
    
    preprocs = pdc_configs(type = "preproc",
                           none = list(),
                           # specify which should consider the shared ones
                           share.config = c("p")
    ),
    
    distances = pdc_configs(type = "distance",#tadpole is ignored
                            partitional = list(
                              sdtw = list(window.size = window_size, norm = c("L2"),
                                          gamma = c(0.001))
                            )
    ),
    
    centroids = pdc_configs(type = "centroid",
                            partitional = list(dba=list(norm = c('L2'),
                                                        znorm = c(FALSE))
                            )
    )
  )
  
  print('done with define configs')
  return(cfgs)
}
