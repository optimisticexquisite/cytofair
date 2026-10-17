# Registered Study 2 mechanism hypotheses and their Holm family.
#
#   H2  CRIC: P1 - P2 accuracy > 0                       (from protocol_lmm.R output)
#   H3  2C:   leakage effect at matched diversity > 0     (factorial accuracy LMM)
#   H4  2B:   context-route difference-in-differences > 0 (transplant LMM)
#   H5  2A:   P1 - P2 gap increases with recoverability   (gradient-reversal sweep)
#
# Each hypothesis is supported only if its Holm-adjusted p < 0.05 AND the
# estimate has the predicted sign. H1 (the SIPaKMeD primary endpoint) is tested
# alone at alpha = 0.05 by protocol_lmm.R and is not part of this family.
#
# Usage:
#   Rscript mechanism_lmm.R FACTORIAL_RUNS TRANSPLANT_RUNS GRL_RUNS CRIC_JSON OUTPUT_JSON

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("usage: mechanism_lmm.R FACTORIAL TRANSPLANT GRL CRIC_JSON OUTPUT_JSON")
control <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

fit_ladder <- function(formulas, data) {
  for (name in names(formulas)) {
    seen <- character()
    candidate <- withCallingHandlers(
      tryCatch(lmer(formulas[[name]], data = data, REML = TRUE, control = control), error = function(e) NULL),
      warning = function(w) {
        seen <<- c(seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    messages <- if (is.null(candidate)) "fit error" else c(candidate@optinfo$conv$lme4$messages, seen)
    failed <- is.null(candidate) ||
      any(grepl("converge", messages, ignore.case = TRUE) & !grepl("singular", messages, ignore.case = TRUE))
    if (!failed) return(list(fit = candidate, used = name, messages = messages))
  }
  list(fit = NULL, used = NA_character_, messages = "all registered formulas failed")
}

coefficient <- function(result, term) {
  if (is.null(result$fit)) return(list(model_used = NA, estimate = NA, p = NA))
  table <- summary(result$fit, ddf = "Satterthwaite")$coefficients
  df <- table[term, "df"]
  list(
    model_used = result$used,
    estimate = table[term, "Estimate"],
    se = table[term, "Std. Error"],
    df = df,
    p = table[term, "Pr(>|t|)"],
    ci95 = table[term, "Estimate"] + c(-1, 1) * qt(0.975, df) * table[term, "Std. Error"],
    singular = isSingular(result$fit),
    variance_components = as.data.frame(VarCorr(result$fit))
  )
}

code_cells <- function(frame) {
  frame$leak <- ifelse(startsWith(frame$cell, "leak"), 0.5, -0.5)
  frame$div <- ifelse(endsWith(frame$cell, "high"), 0.5, -0.5)
  frame$architecture <- factor(frame$architecture)
  frame$seed <- factor(frame$seed)
  frame$training_set <- interaction(frame$seed, frame$cell, drop = TRUE)
  frame
}

out <- list(r_version = R.version.string)

# ---------------- H3: leakage at matched diversity (2C) ----------------
factorial <- read.csv(args[[1]], stringsAsFactors = FALSE)
factorial$cell <- factorial$protocol
factorial <- code_cells(factorial)
factorial$value <- 100 * factorial$accuracy
if (nrow(factorial) != 4 * 10 * 4) stop(sprintf("2C matrix has %d rows, expected 160", nrow(factorial)))
h3 <- fit_ladder(list(
  full = value ~ leak * div + (1 + leak | architecture) + (1 | seed) + (1 | training_set),
  uncorrelated = value ~ leak * div + (1 + leak || architecture) + (1 | seed) + (1 | training_set),
  intercepts = value ~ leak * div + (1 | architecture) + (1 | seed) + (1 | training_set)
), factorial)
out$H3 <- coefficient(h3, "leak")
out$H3_interaction_exploratory <- coefficient(h3, "leak:div")
clean <- subset(factorial, leak < 0)
diversity <- fit_ladder(list(
  full = value ~ div + (1 + div | architecture) + (1 | seed) + (1 | training_set),
  intercepts = value ~ div + (1 | architecture) + (1 | seed) + (1 | training_set)
), clean)
out$diversity_effect_clean_secondary <- coefficient(diversity, "div")
out$cell_means <- aggregate(value ~ cell + architecture, data = factorial, FUN = mean)

# ---------------- H4: context-route difference-in-differences (2B) ----------------
transplant <- read.csv(args[[2]], stringsAsFactors = FALSE)
transplant <- code_cells(transplant)
all_recipients <- subset(transplant, subset == "all")
if (nrow(all_recipients) != 160) stop(sprintf("2B table has %d runs, expected 160", nrow(all_recipients)))
all_recipients$route <- 100 * (all_recipients$acc_sibling - all_recipients$acc_unseen_same_class)
h4 <- fit_ladder(list(
  full = route ~ leak + div + (1 + leak | architecture) + (1 | seed) + (1 | training_set),
  intercepts = route ~ leak + div + (1 | architecture) + (1 | seed) + (1 | training_set)
), all_recipients)
out$H4 <- coefficient(h4, "leak")
all_recipients$original <- 100 * all_recipients$acc_original
leak_gap <- fit_ladder(list(
  intercepts = original ~ leak + div + (1 | architecture) + (1 | seed) + (1 | training_set)
), all_recipients)
out$H4_leak_effect_on_original_images <- coefficient(leak_gap, "leak")
shift <- fit_ladder(list(
  intercepts = I(100 * donor_probability_shift) ~ leak + div + (1 | architecture) + (1 | seed)
), all_recipients)
out$background_class_information_secondary <- coefficient(shift, "(Intercept)")
restricted <- subset(transplant, subset == "background_ge_0.30")
restricted$route <- 100 * (restricted$acc_sibling - restricted$acc_unseen_same_class)
out$H4_background_ge_0.30_secondary <- coefficient(fit_ladder(list(
  intercepts = route ~ leak + div + (1 | architecture) + (1 | seed) + (1 | training_set)
), restricted), "leak")
out$transplant_condition_means <- aggregate(
  cbind(acc_original, acc_self, acc_sibling, acc_unseen_same_class, acc_unseen_other_class) ~ cell,
  data = all_recipients, FUN = mean
)

# ---------------- H5: fingerprint suppression dose-response (2A) ----------------
grl <- read.csv(args[[3]], stringsAsFactors = FALSE)
grl$R <- 100 * grl$within_class_accuracy
grl$floor <- 100 * grl$within_class_floor
grl$value <- 100 * grl$accuracy
means <- aggregate(cbind(value, R, floor) ~ architecture + protocol + adversary_lambda, data = grl, FUN = mean)
out$grl_sweep_means <- means
pairs <- merge(
  subset(grl, protocol == "P1")[, c("architecture", "seed", "adversary_lambda", "value", "R")],
  subset(grl, protocol == "P2")[, c("architecture", "seed", "adversary_lambda", "value", "R")],
  by = c("architecture", "seed", "adversary_lambda"), suffixes = c("_p1", "_p2")
)
pairs$gap <- pairs$value_p1 - pairs$value_p2
pairs$R_pair <- (pairs$R_p1 + pairs$R_p2) / 2
eligible <- list()
checks <- list()
for (arch_name in sort(unique(pairs$architecture))) {
  p2 <- means[means$architecture == arch_name & means$protocol == "P2", ]
  baseline <- p2$value[p2$adversary_lambda == 0]
  # Non-degenerate regime: mean P2 accuracy within 1 pp of the lambda = 0 baseline.
  lambdas <- sort(p2$adversary_lambda[p2$value >= baseline - 1.0])
  lambda_star <- max(lambdas)
  arch_pairs <- pairs[pairs$architecture == arch_name, ]
  floor_value <- mean(grl$floor[grl$architecture == arch_name])
  r0 <- mean(arch_pairs$R_pair[arch_pairs$adversary_lambda == 0])
  r_star <- mean(arch_pairs$R_pair[arch_pairs$adversary_lambda == lambda_star])
  # Manipulation check: suppression must remove at least a quarter of the
  # above-floor within-class recoverability without leaving the regime.
  passed <- r_star <= r0 - 0.25 * (r0 - floor_value)
  checks[[arch_name]] <- list(
    non_degenerate_lambdas = lambdas, lambda_star = lambda_star, R_baseline = r0, R_star = r_star,
    within_class_floor = floor_value, manipulation_check_passed = passed
  )
  if (passed) eligible[[arch_name]] <- arch_pairs[arch_pairs$adversary_lambda %in% lambdas, ]
}
out$H5_manipulation_checks <- checks
if (length(eligible) == 0) {
  out$H5 <- list(status = "untestable: fingerprint suppression failed its manipulation check", p = NA, estimate = NA)
} else {
  h5_data <- do.call(rbind, eligible)
  h5_data$seed <- factor(h5_data$seed)
  h5_data$architecture <- factor(h5_data$architecture)
  formula <- if (nlevels(h5_data$architecture) > 1) gap ~ architecture + R_pair + (1 | seed) else gap ~ R_pair + (1 | seed)
  out$H5 <- coefficient(fit_ladder(list(fixed_architecture = formula), h5_data), "R_pair")
  out$H5$architectures <- levels(h5_data$architecture)
}

# ---------------- H2 from the CRIC protocol model, then Holm ----------------
cric <- fromJSON(args[[4]])
out$H2 <- list(estimate = cric$primary_estimate, p = cric$primary_p, source = cric$primary_source)
p_values <- c(H2 = out$H2$p, H3 = out$H3$p, H4 = out$H4$p, H5 = out$H5$p)
estimates <- c(H2 = out$H2$estimate, H3 = out$H3$estimate, H4 = out$H4$estimate, H5 = out$H5$estimate)
testable <- !is.na(p_values)
adjusted <- rep(NA_real_, length(p_values))
names(adjusted) <- names(p_values)
adjusted[testable] <- p.adjust(p_values[testable], method = "holm")
out$holm <- lapply(names(p_values), function(h) list(
  hypothesis = h, p = unname(p_values[h]), p_holm = unname(adjusted[h]), estimate = unname(estimates[h]),
  supported = isTRUE(adjusted[h] < 0.05 && estimates[h] > 0)
))
write_json(out, args[[5]], auto_unbox = TRUE, pretty = TRUE, digits = NA)
for (item in out$holm) {
  cat(sprintf("%s estimate=%.3f p=%.4g p_holm=%.4g supported=%s\n",
              item$hypothesis, item$estimate, item$p, item$p_holm, item$supported))
}
