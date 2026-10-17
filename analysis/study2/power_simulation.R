# Simulation-based power analysis for the Study 2 omnibus protocol effect.
#
# Pilot estimates come from the frozen five-seed SIPaKMeD matrix (Study 1). The
# confirmatory model is fitted to every simulated data set exactly as it will be
# fitted to the real data, so the power estimate includes the Satterthwaite
# degrees-of-freedom penalty for having few architectures and splits.
#
# Usage:
#   Rscript power_simulation.R PILOT_CSV OUTPUT_PREFIX NSIM CORES

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
  library(parallel)
  library(MASS)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: power_simulation.R PILOT_CSV OUTPUT_PREFIX NSIM CORES")
}
pilot_path <- args[[1]]
out_prefix <- args[[2]]
nsim <- as.integer(args[[3]])
cores <- as.integer(args[[4]])

confirmatory_formula <- accuracy ~ x + (1 + x | architecture) + (1 | seed) + (1 | split)

prepare <- function(frame) {
  frame <- subset(frame, protocol %in% c("P1", "P2"))
  # Centred contrast: the fixed effect of x is exactly the P1 - P2 difference and
  # the architecture intercept is the architecture's mean level.
  frame$x <- ifelse(frame$protocol == "P1", 0.5, -0.5)
  frame$architecture <- factor(frame$architecture)
  frame$seed <- factor(frame$seed)
  frame$split <- interaction(frame$seed, frame$protocol, drop = TRUE)
  frame
}

pilot <- prepare(read.csv(pilot_path))

# Homoscedastic pilot fit (the confirmatory specification).
fit_h <- lmer(confirmatory_formula, data = pilot, REML = TRUE)
# Heteroscedastic split variance for the data-generating model: P2 test sets are
# clustered within roughly 190 parents, so their split-to-split variance is larger.
pilot$is_p1 <- as.numeric(pilot$protocol == "P1")
pilot$is_p2 <- as.numeric(pilot$protocol == "P2")
fit_g <- lmer(
  accuracy ~ x + (1 + x | architecture) + (1 | seed) + (0 + is_p1 | split) + (0 + is_p2 | split),
  data = pilot, REML = TRUE
)

vc_h <- as.data.frame(VarCorr(fit_h))
vc_g <- as.data.frame(VarCorr(fit_g))
pick <- function(vc, grp_pattern, var1, var2 = NA) {
  # lme4 de-duplicates repeated grouping names ("split", "split.1"), so split
  # components are matched by their term rather than by their position.
  row <- vc[grepl(grp_pattern, vc$grp) & vc$var1 %in% var1 &
    (if (is.na(var2)) is.na(vc$var2) else vc$var2 %in% var2), ]
  if (nrow(row) != 1) stop(sprintf("variance component %s/%s not unique", grp_pattern, var1))
  row
}
arch_int <- pick(vc_g, "^architecture$", "(Intercept)")$sdcor
arch_slope <- pick(vc_g, "^architecture$", "x")$sdcor
pilot_rho <- pick(vc_g, "^architecture$", "(Intercept)", "x")$sdcor
seed_sd <- pick(vc_g, "^seed$", "(Intercept)")$sdcor
split_p1 <- pick(vc_g, "^split", "is_p1")$sdcor
split_p2 <- pick(vc_g, "^split", "is_p2")$sdcor
resid_sd <- pick(vc_g, "^Residual$", NA)$sdcor
# Six architectures put the intercept-slope correlation on its -1 boundary. A
# degenerate covariance is not a credible data-generating model, so the
# simulation uses independent architecture intercepts and slopes; the pilot
# estimate is still recorded below.
arch_rho <- 0

coef_h <- summary(fit_h)$coefficients
pilot_pars <- list(
  mu = unname(fixef(fit_g)[["(Intercept)"]]),
  pilot_effect = unname(fixef(fit_h)[["x"]]),
  pilot_effect_se = coef_h["x", "Std. Error"],
  pilot_effect_df = coef_h["x", "df"],
  pilot_effect_p = coef_h["x", "Pr(>|t|)"],
  sd_int = arch_int,
  sd_slope = arch_slope,
  rho = arch_rho,
  pilot_rho = pilot_rho,
  sd_seed = seed_sd,
  sd_split_p1 = split_p1,
  sd_split_p2 = split_p2,
  sd_resid = resid_sd,
  homoscedastic_components = vc_h,
  heteroscedastic_components = vc_g,
  pilot_singular_homoscedastic = isSingular(fit_h),
  pilot_singular_heteroscedastic = isSingular(fit_g),
  n_pilot_rows = nrow(pilot)
)

simulate_once <- function(S, A, delta, pars, scale, rng_seed) {
  set.seed(rng_seed)
  d <- expand.grid(
    architecture = factor(seq_len(A)),
    seed = factor(seq_len(S)),
    protocol = c("P1", "P2"),
    KEEP.OUT.ATTRS = FALSE
  )
  d$x <- ifelse(d$protocol == "P1", 0.5, -0.5)
  d$split <- interaction(d$seed, d$protocol, drop = TRUE)
  sd_int <- pars$sd_int * scale
  sd_slope <- pars$sd_slope * scale
  covariance <- matrix(
    c(sd_int^2, pars$rho * sd_int * sd_slope, pars$rho * sd_int * sd_slope, sd_slope^2), 2
  )
  arch_effects <- mvrnorm(A, c(0, 0), covariance)
  if (A == 1) arch_effects <- matrix(arch_effects, nrow = 1)
  seed_effects <- rnorm(S, 0, pars$sd_seed * scale)
  split_p1 <- rnorm(S, 0, pars$sd_split_p1 * scale)
  split_p2 <- rnorm(S, 0, pars$sd_split_p2 * scale)
  ai <- as.integer(d$architecture)
  si <- as.integer(d$seed)
  split_effect <- ifelse(d$protocol == "P1", split_p1[si], split_p2[si])
  d$accuracy <- pars$mu + delta * d$x + arch_effects[ai, 1] + arch_effects[ai, 2] * d$x +
    seed_effects[si] + split_effect + rnorm(nrow(d), 0, pars$sd_resid * scale)
  fit <- tryCatch(
    suppressMessages(suppressWarnings(lmer(confirmatory_formula, data = d, REML = TRUE))),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    return(c(estimate = NA, se = NA, df = NA, p = NA, singular = NA, converged = 0))
  }
  coefs <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
  if (is.null(coefs) || !("df" %in% colnames(coefs))) {
    return(c(estimate = NA, se = NA, df = NA, p = NA, singular = NA, converged = 0))
  }
  c(
    estimate = coefs["x", "Estimate"],
    se = coefs["x", "Std. Error"],
    df = coefs["x", "df"],
    p = coefs["x", "Pr(>|t|)"],
    singular = as.numeric(isSingular(fit)),
    converged = 1
  )
}

grid <- expand.grid(
  seeds = c(10, 15, 20, 25, 30),
  architectures = c(6, 8, 10, 12),
  delta = c(0.5, 0.8, 1.0),
  scale = c(1.0, 1.5),
  KEEP.OUT.ATTRS = FALSE
)
# The inflated-variance sensitivity is only needed for the primary 0.8-pp target.
grid <- grid[!(grid$scale == 1.5 & grid$delta != 0.8), ]

jobs <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  data.frame(cell = i, sim = seq_len(nsim))
}))
run_job <- function(j) {
  g <- grid[jobs$cell[j], ]
  # Deterministic, cell-specific random stream.
  rng_seed <- 20261018L + 100000L * jobs$cell[j] + jobs$sim[j]
  out <- simulate_once(g$seeds, g$architectures, g$delta, pilot_pars, g$scale, rng_seed)
  c(cell = jobs$cell[j], sim = jobs$sim[j], out)
}
RNGkind("L'Ecuyer-CMRG")
results <- mclapply(seq_len(nrow(jobs)), run_job, mc.cores = cores, mc.preschedule = TRUE)
results <- as.data.frame(do.call(rbind, results))

summaries <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  r <- results[results$cell == i & results$converged == 1, ]
  reject <- r$p < 0.05 & r$estimate > 0
  power <- mean(reject)
  data.frame(
    grid[i, ],
    n_sim = nsim,
    n_converged = nrow(r),
    power = power,
    power_mcse = sqrt(power * (1 - power) / max(1, nrow(r))),
    mean_estimate = mean(r$estimate),
    mean_se = mean(r$se),
    median_df = median(r$df),
    singular_fraction = mean(r$singular)
  )
}))

write.csv(results, paste0(out_prefix, "_simulations.csv"), row.names = FALSE)
write.csv(summaries, paste0(out_prefix, "_power.csv"), row.names = FALSE)
write_json(
  list(
    pilot = pilot_pars[setdiff(names(pilot_pars), c("homoscedastic_components", "heteroscedastic_components"))],
    homoscedastic_components = vc_h,
    heteroscedastic_components = vc_g,
    nsim = nsim,
    formula = deparse(confirmatory_formula),
    r_version = R.version.string,
    lme4_version = as.character(packageVersion("lme4")),
    lmerTest_version = as.character(packageVersion("lmerTest"))
  ),
  paste0(out_prefix, "_pilot.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = NA
)
print(summaries)
