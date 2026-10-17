# Confirmatory protocol contrast for Study 2 (H1: SIPaKMeD, H2: CRIC).
#
#   value ~ x + (1 + x | architecture) + (1 | seed) + (1 | split)
#
# x = +0.5 for the left protocol and -0.5 for the right protocol, so the fixed
# effect is exactly left - right in percentage points. `split` is the
# seed-by-protocol partition shared by every architecture trained on it; without
# it, twelve architectures evaluated on one test set would be counted as twelve
# independent test sets.
#
# Usage:
#   Rscript protocol_lmm.R RUNS_CSV OUTPUT_JSON VALUE LEFT RIGHT EXPECTED_ARCHS EXPECTED_SEEDS [BOOT]
#
# The script fails closed if any registered architecture-seed-protocol cell is
# missing or duplicated.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("usage: protocol_lmm.R RUNS_CSV OUTPUT_JSON VALUE LEFT RIGHT EXPECTED_ARCHS EXPECTED_SEEDS [BOOT]")
}
runs_path <- args[[1]]
output_path <- args[[2]]
value <- args[[3]]
left <- args[[4]]
right <- args[[5]]
expected_archs <- as.integer(args[[6]])
expected_seeds <- as.integer(args[[7]])
n_boot <- if (length(args) >= 8) as.integer(args[[8]]) else 10000L

runs <- read.csv(runs_path, stringsAsFactors = FALSE)
data <- subset(runs, protocol %in% c(left, right))
data$value <- 100 * data[[value]]
key <- paste(data$architecture, data$protocol, data$seed)
if (any(duplicated(key))) stop("duplicate architecture/protocol/seed rows")
if (length(unique(data$architecture)) != expected_archs) stop("unexpected architecture count")
if (length(unique(data$seed)) != expected_seeds) stop("unexpected seed count")
if (nrow(data) != 2 * expected_archs * expected_seeds) {
  stop(sprintf("incomplete matrix: %d rows, expected %d", nrow(data), 2 * expected_archs * expected_seeds))
}
data$x <- ifelse(data$protocol == left, 0.5, -0.5)
data$architecture <- factor(data$architecture)
data$seed <- factor(data$seed)
data$split <- interaction(data$seed, data$protocol, drop = TRUE)

control <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# Pre-registered simplification ladder, applied only on convergence failure
# (a singular fit is a boundary estimate, not a failure).
ladder <- list(
  full = value ~ x + (1 + x | architecture) + (1 | seed) + (1 | split),
  uncorrelated = value ~ x + (1 + x || architecture) + (1 | seed) + (1 | split),
  no_seed = value ~ x + (1 + x || architecture) + (1 | split)
)
fit <- NULL
used <- NA_character_
messages <- list()
for (name in names(ladder)) {
  warnings_seen <- character()
  candidate <- withCallingHandlers(
    tryCatch(lmer(ladder[[name]], data = data, REML = TRUE, control = control), error = function(e) NULL),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  convergence <- if (is.null(candidate)) "fit error" else candidate@optinfo$conv$lme4$messages
  failed <- is.null(candidate) || any(grepl("converge", c(convergence, warnings_seen), ignore.case = TRUE) &
    !grepl("singular", c(convergence, warnings_seen), ignore.case = TRUE))
  messages[[name]] <- c(convergence, warnings_seen)
  if (!failed) {
    fit <- candidate
    used <- name
    break
  }
}

result <- list(
  value = value,
  left = left,
  right = right,
  n_rows = nrow(data),
  n_architectures = expected_archs,
  n_seeds = expected_seeds,
  model_messages = messages,
  model_used = used,
  r_version = R.version.string,
  lme4_version = as.character(packageVersion("lme4")),
  lmerTest_version = as.character(packageVersion("lmerTest"))
)

raw_difference <- with(
  merge(
    subset(data, protocol == left)[, c("architecture", "seed", "value")],
    subset(data, protocol == right)[, c("architecture", "seed", "value")],
    by = c("architecture", "seed"), suffixes = c("_left", "_right")
  ),
  data.frame(architecture, seed, d = value_left - value_right)
)
result$raw_mean_difference <- mean(raw_difference$d)

if (!is.null(fit)) {
  satterthwaite <- summary(fit, ddf = "Satterthwaite")$coefficients
  kenward <- tryCatch(summary(fit, ddf = "Kenward-Roger")$coefficients, error = function(e) NULL)
  estimate <- satterthwaite["x", "Estimate"]
  se <- satterthwaite["x", "Std. Error"]
  df <- satterthwaite["x", "df"]
  result$estimate <- estimate
  result$se <- se
  result$df_satterthwaite <- df
  result$t <- satterthwaite["x", "t value"]
  result$p_satterthwaite <- satterthwaite["x", "Pr(>|t|)"]
  result$ci95 <- estimate + c(-1, 1) * qt(0.975, df) * se
  if (!is.null(kenward)) {
    result$df_kenward_roger <- kenward["x", "df"]
    result$p_kenward_roger <- kenward["x", "Pr(>|t|)"]
  }
  result$singular <- isSingular(fit)
  result$variance_components <- as.data.frame(VarCorr(fit))
  residuals <- resid(fit)
  centred <- residuals - mean(residuals)
  skewness <- mean(centred^3) / mean(centred^2)^1.5
  kurtosis <- mean(centred^4) / mean(centred^2)^2 - 3
  slopes <- ranef(fit)$architecture
  result$assumptions <- list(
    residual_skewness = skewness,
    residual_excess_kurtosis = kurtosis,
    residual_shapiro_p = shapiro.test(residuals)$p.value,
    architecture_slope_blups = if ("x" %in% names(slopes)) setNames(slopes$x, rownames(slopes)) else NULL,
    residual_sd_by_protocol = tapply(residuals, data$protocol, sd)
  )
  result$assumptions_violated <- abs(skewness) > 1 || kurtosis > 3
}

# Two-way cluster bootstrap over architectures and seeds (always reported; the
# registered primary result if the model fails or its assumptions are violated).
set.seed(20261018)
archs <- levels(data$architecture)
seeds <- levels(data$seed)
mean_by <- tapply(raw_difference$d, list(raw_difference$architecture, raw_difference$seed), mean)
boot <- replicate(n_boot, {
  a <- sample(archs, replace = TRUE)
  s <- sample(seeds, replace = TRUE)
  mean(mean_by[a, s])
})
result$cluster_bootstrap <- list(
  replicates = n_boot,
  ci95 = unname(quantile(boot, c(0.025, 0.975))),
  p_two_sided = min(1, 2 * min(mean(boot <= 0), mean(boot >= 0)))
)

# Seed-level sign-flip test (conditions on the architecture set).
seed_difference <- tapply(raw_difference$d, raw_difference$seed, mean)
observed <- mean(seed_difference)
flips <- 2^length(seed_difference)
if (flips <= 2^20) {
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), length(seed_difference))))
  null <- as.vector(signs %*% seed_difference) / length(seed_difference)
} else {
  null <- replicate(200000, mean(sample(c(-1, 1), length(seed_difference), replace = TRUE) * seed_difference))
}
result$sign_flip <- list(
  statistic = observed,
  permutations = length(null),
  p_two_sided = mean(abs(null) >= abs(observed) - 1e-12)
)

primary_source <- if (is.null(fit) || isTRUE(result$assumptions_violated)) "cluster_bootstrap" else "lmm"
result$primary_source <- primary_source
if (primary_source == "lmm") {
  result$primary_p <- result$p_satterthwaite
  result$primary_ci95 <- result$ci95
  result$primary_estimate <- result$estimate
} else {
  result$primary_p <- result$cluster_bootstrap$p_two_sided
  result$primary_ci95 <- result$cluster_bootstrap$ci95
  result$primary_estimate <- result$raw_mean_difference
}
result$supported <- result$primary_p < 0.05 && result$primary_estimate > 0

# Binomial GLMM sensitivity on correct counts (accuracy endpoints only).
if (value == "accuracy" && "test_crops" %in% names(data)) {
  data$correct <- round(data$value / 100 * data$test_crops)
  data$observation <- factor(seq_len(nrow(data)))
  glmm <- tryCatch(
    suppressWarnings(glmer(
      cbind(correct, test_crops - correct) ~ x + (1 + x || architecture) + (1 | seed) + (1 | split) + (1 | observation),
      data = data, family = binomial, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
    )),
    error = function(e) NULL
  )
  if (!is.null(glmm)) {
    coefficients <- summary(glmm)$coefficients
    result$binomial_glmm <- list(
      log_odds_ratio = coefficients["x", "Estimate"],
      se = coefficients["x", "Std. Error"],
      p_wald = coefficients["x", "Pr(>|z|)"]
    )
  }
}

per_architecture <- do.call(rbind, lapply(split(raw_difference, raw_difference$architecture), function(g) {
  test <- t.test(g$d)
  data.frame(
    architecture = as.character(g$architecture[1]),
    mean_difference = mean(g$d),
    sd_difference = sd(g$d),
    ci_low = test$conf.int[1],
    ci_high = test$conf.int[2],
    n = nrow(g)
  )
}))
result$per_architecture_descriptive <- per_architecture

write_json(result, output_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat(sprintf(
  "%s %s-%s: estimate=%.3f pp, p=%.4g (%s), supported=%s\n",
  value, left, right, result$primary_estimate, result$primary_p, primary_source, result$supported
))
