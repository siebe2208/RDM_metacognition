functions {


  // psychometric function to get the probability of responding (1)
  real psycho_ACC(real x, real alpha, real beta, real lapse){
    return (lapse + (1-2*lapse) * inv_logit(beta * (x - alpha)));
   }

  // entropy of a binary decision variable
  real entropy(real p){
    return(-p * log(p) - (1-p) * log(1-p));
  }

 // cummulative ordered beta distribution density
  real ord_beta_reg_cdf(real y, real mu, real phi, real cutzero, real cutone) {

    vector[2] thresh;
    thresh[1] = cutzero;
    thresh[2] = cutzero + exp(cutone);

    real p0 = 1-inv_logit(mu - thresh[1]);

    real p_m = (inv_logit(mu - thresh[1])-inv_logit(mu - thresh[2]))  * beta_cdf(y | exp(log_inv_logit(mu) + log(phi)), exp(log1m_inv_logit(mu) + log(phi)));



    if (y < 0) {
      return 0;
    } else if (y == 0) {
      return p0;
    } else if (y == 1) {
      return 1-(1e-12);
    } else {
      return (p0 + p_m);
    }
  }

 // ordered beta distribution density
  real ord_beta_reg_lpdf(real y, real mu, real phi, real cutzero, real cutone) {

    vector[2] thresh;
    thresh[1] = cutzero;
    thresh[2] = cutzero + exp(cutone);

  if(y==0) {
      return log1m_inv_logit(mu - thresh[1]);
    } else if(y==1) {
      return log_inv_logit(mu  - thresh[2]);
    } else {
      return log_diff_exp(log_inv_logit(mu - thresh[1]), log_inv_logit(mu - thresh[2])) +
                beta_lpdf(y|exp(log_inv_logit(mu) + log(phi)),exp(log1m_inv_logit(mu) + log(phi)));
    }
  }


  // priors for the cutpoints so that they are ordered
  real induced_dirichlet_lpdf(real nocut, vector alpha, real phi, int cutnum, real cut1, real cut2) {
    int K = num_elements(alpha);
    vector[K-1] c = [cut1, cut1 + exp(cut2)]';
    vector[K - 1] sigma = inv_logit(phi - c);
    vector[K] p;
    matrix[K, K] J = rep_matrix(0, K, K);

    if(cutnum==1) {

    // Induced ordinal probabilities
    p[1] = 1 - sigma[1];
    for (k in 2:(K - 1))
      p[k] = sigma[k - 1] - sigma[k];
    p[K] = sigma[K - 1];

    // Baseline column of Jacobian
    for (k in 1:K) J[k, 1] = 1;

    // Diagonal entries of Jacobian
    for (k in 2:K) {
      real rho = sigma[k - 1] * (1 - sigma[k - 1]);
      J[k, k] = - rho;
      J[k - 1, k] = rho;
    }

    // divide in half for the two cutpoints

    // don't forget the ordered transformation

      return   dirichlet_lpdf(p | alpha)
           + log_determinant(J) + cut2;
    } else {
      return(0);
    }
  }


 // translate the probability of being correct into the two curves of being correct or being wrong
  real get_conf(real ACC, real theta, real x, real alpha){
  if(ACC == 1 && x > alpha){
    return(theta);
  }else if(ACC == 1 && x < alpha){
    return(1-theta);
  }else if(ACC == 0 && x > alpha){
    return(1-theta);
  }else if(ACC == 0 && x < alpha){
    return(theta);
  }else{
    return(0);
  }
}

  // get the probability of being correct from probability of responding "(1)"
  real get_prob_cor(real theta, real x){
  if(x > 0){
    return(theta);
  }else if(x < 0){
    return(1-theta);
  }else{
    return(0);
  }

}
}


data {
  // number of data points
  int<lower=0> N;

  // binary responses (here correct or incorrect)
  array[N] int binom_y;
  //Response times
  vector[N] RT;
  // confidence ratings
  vector[N] Conf;

  // stimulus value
  vector[N] X;

  // the minimum response time
  real minRT;

  // the accuracy of the subject at trial.
  vector[N] ACC;

}

transformed data{
  // number of parameters for each subject:
  int P = 9;
}

parameters {
  // number of group parameters
  vector[P] gm;

  // cutpoint for responding 0
  real c0;
  //cutpoint for responding 1
  real c11;
    // on-decision time parameter that is between 0 and their lowest response time. As the non-decision time can't be less than the lowest response time.
  real<lower=0, upper = minRT> rt_ndt;

}



transformed parameters{

  // extracting the subject level paramters
  //psychometric function parametesr
  real alpha = (gm[1]);
  real beta = (gm[2]);
  real lapse = inv_logit(gm[3]) / 2;

  // response time distribution
  real rt_int = gm[4];
  real rt_slope = gm[5];
  real rt_prec = exp(gm[6]);

  //confidence parameters.
  real conf_prec = exp(gm[7]);
  real meta_un = gm[8];
  real meta_bias = gm[9];



 //trial by trial predictions of the model for:
 //entropy
  vector[N] entropy_t;
  //mean confidence
  vector[N] conf_mu;
  //probability of responding 1
  vector[N] theta;
  // probability of responding 1 (from the perspective of confidence), i.e. with meta cognitive noise:
  vector[N] theta_conf;


  // this is filling up these above trial by trial predictions with the model.
  // it loops through all trials and puts in the model prediction at that trial for that subject notice the S_id[n] for all subject level parameters.
  for (n in 1:N) {
  theta[n] = psycho_ACC(X[n], (alpha), exp(beta), lapse);

  entropy_t[n] = entropy(psycho_ACC(X[n], (alpha), exp(beta), lapse));

  theta_conf[n] = psycho_ACC(X[n], (alpha), exp(beta + meta_un), lapse);

  conf_mu[n] = get_conf(ACC[n],theta_conf[n], X[n], alpha);
  }

}
model {

  // Priors

  //psychometric functiuon parameters
  gm[1] ~ normal(0,5);
  gm[2] ~ normal(-2,2);
  gm[3] ~ normal(-4,2);

  // RT and confidence parameters
  gm[4:9] ~ normal(0,2);

  // prior on the non-decision time.
  rt_ndt ~ normal(0.3,0.05);


  // calculating the uniform variables for the copula bit, by using the cummulative marginal distribution to get a uniform variable (based on the model), for each of the three response variables (B,RT,C)
  for (n in 1:N) {


    // likelihood of the binary responses

    target += binomial_lpmf(binom_y[n] | 1, get_prob_cor(theta[n], X[n]));

    // here we then add the data (RT and Conf[n]) to the model that we have (i.e. telling stan that this is what need to be optimized)
    // first the likeliood for the response times
    target += lognormal_lpdf(RT[n] - rt_ndt | rt_int + rt_slope * entropy_t[n], rt_prec);

    // then the likelihood for the confidnnce ratings (notice the meta noise is adding in logit space)
    target += ord_beta_reg_lpdf(Conf[n] | logit(conf_mu[n])+ meta_bias, conf_prec, c0, c11);

  }

    // this is the prior for the cutpoints (they are a bit special as they need to be ordered)
    c0 ~ induced_dirichlet([1,10,1]', 0, 1, c0, c11);
    c11 ~ induced_dirichlet([1,10,1]', 0, 2, c0, c11);


}


// all of this are just extract computations for convinence afters (all this can be calculated from the above in R afterwards.)
generated quantities {

  real c1 = c0 + exp(c11);
  real rho_p_rt;
  real rho_p_conf;
  real rho_rt_conf;

  vector[N] log_lik_bin = rep_vector(0,N);
  vector[N] log_lik_rt = rep_vector(0,N);
  vector[N] log_lik_conf = rep_vector(0,N);
  vector[N] log_lik = rep_vector(0,N);





  for(n in 1:N){
    log_lik_bin[n] = binomial_lpmf(binom_y[n] | 1, get_prob_cor(theta[n], X[n]));
    log_lik_rt[n] = lognormal_lpdf(RT[n] - rt_ndt | rt_int + rt_slope * entropy_t[n], rt_prec);
    log_lik_conf[n] = ord_beta_reg_lpdf(Conf[n] | logit(conf_mu[n]) + meta_bias, conf_prec, c0, c11);
    log_lik[n] = log_lik_bin[n] + log_lik_rt[n] + log_lik_conf[n];
  }



}
