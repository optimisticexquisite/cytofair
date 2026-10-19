# Registered Amendment 8 follow-up hypotheses and their Holm family.
#
#   F1  CRIC: slide-context pull on discordant cells > 0            (cric_context.py output)
#   F2  2C checkpoints: test-time stain transfer reduces the leak effect
#   F3  carrier factorial: the leak effect survives background masking (> 0)
#   F4  carrier factorial: colour augmentation reduces the leak effect
#   F5  Phase 1 checkpoints: LEACE erasure of within-class parent identity reduces the P1 - P2 gap
#   F6  stain normalisation strength reduces the P1 - P2 gap
#   F7  probe-matched linear adversary: P1 - P2 gap increases with recoverability R
#   F8  pooled over F5-F7 manipulations: P1 - P2 gap increases with R
#
# Every estimate is oriented so that the predicted direction is positive. Each is
# supported only if its Holm-adjusted p < 0.05 and its estimate is positive.
#
# Usage: Rscript followup_lmm.R INPUT_DIR CRIC_JSON OUTPUT_JSON

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: followup_lmm.R INPUT_DIR CRIC_JSON OUTPUT_JSON")
input <- args[[1]]
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

coefficient <- function(result, term, sign = 1) {
  if (is.null(result$fit)) return(list(model_used = NA, estimate = NA, p = NA))
  table <- summary(result$fit, ddf = "Satterthwaite")$coefficients
  df <- table[term, "df"]
  ci <- sign * (table[term, "Estimate"] + c(-1, 1) * qt(0.975, df) * table[term, "Std. Error"])
  list(
    model_used = result$used,
    term = term,
    sign = sign,
    estimate = sign * table[term, "Estimate"],
    se = table[term, "Std. Error"],
    df = df,
    p = table[term, "Pr(>|t|)"],
    ci95 = sort(ci),
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

expect_rows <- function(frame, n, label) {
  if (nrow(frame) != n) stop(sprintf("%s has %d rows, expected %d", label, nrow(frame), n))
}

leak_formulas <- list(
  full = value ~ leak * div + (1 + leak | architecture) + (1 | seed) + (1 | training_set),
  uncorrelated = value ~ leak * div + (1 + leak || architecture) + (1 | seed) + (1 | training_set),
  intercepts = value ~ leak * div + (1 | architecture) + (1 | seed) + (1 | training_set)
)

out <- list(r_version = R.version.string)

# ---------------- F2: test-time carrier ablations on the 2C checkpoints ----------------
ablations <- code_cells(read.csv(file.path(input, "ablation_runs.csv"), stringsAsFactors = FALSE))
expect_rows(ablations, 160 * 6, "ablation table")
ablations$value <- 100 * ablations$accuracy
ablations$run <- interaction(ablations$architecture, ablations$seed, ablations$cell, drop = TRUE)
ablation_reduction <- function(name) {
  data <- subset(ablations, ablation %in% c("original", name))
  data$ablated <- ifelse(data$ablation == name, 0.5, -0.5)
  # leak:ablated is (leak effect under the ablation) - (leak effect on originals);
  # its negative is the reduction, predicted positive.
  coefficient(fit_ladder(list(
    full = value ~ leak * ablated + div + (1 + leak | architecture) + (1 | seed) + (1 | training_set) + (1 | run),
    intercepts = value ~ leak * ablated + div + (1 | architecture) + (1 | seed) + (1 | training_set) + (1 | run)
  ), data), "leak:ablated", sign = -1)
}
out$F2 <- ablation_reduction("stain_transfer")
out$ablation_reductions_secondary <- lapply(
  setNames(nm = c("grey", "low_pass", "illumination_flat", "background_grey")), ablation_reduction
)
out$leak_effect_by_ablation_secondary <- lapply(setNames(nm = sort(unique(ablations$ablation))), function(name) {
  coefficient(fit_ladder(list(
    intercepts = value ~ leak + div + (1 | architecture) + (1 | seed) + (1 | training_set)
  ), subset(ablations, ablation == name)), "leak")
})
out$ablation_cell_means <- aggregate(value ~ ablation + cell, data = ablations, FUN = mean)
out$original_prediction_agreement <- as.list(unclass(summary(ablations$original_prediction_agreement)))

cams <- read.csv(file.path(input, "gradcam_runs.csv"), stringsAsFactors = FALSE)
if (nrow(cams) > 0) {
  cams <- code_cells(cams)
  cams$value <- 100 * cams$background_attribution_fraction
  cams$excess <- 100 * (cams$background_attribution_fraction - cams$background_pixel_fraction)
  out$gradcam_descriptive <- list(
    leak_effect_on_background_attribution = coefficient(fit_ladder(list(
      intercepts = value ~ leak + div + (1 | architecture) + (1 | seed) + (1 | training_set)
    ), cams), "leak"),
    means = aggregate(cbind(value, excess, random_head_spearman) ~ cell + architecture, data = cams, FUN = mean)
  )
}

# ---------------- F3 / F4: carrier removal during training ----------------
carrier <- code_cells(read.csv(file.path(input, "carrier_runs.csv"), stringsAsFactors = FALSE))
expect_rows(carrier, 3 * 4 * 4 * 10, "carrier table")
carrier$value <- 100 * carrier$accuracy
masked <- subset(carrier, condition == "masked")
out$F3 <- coefficient(fit_ladder(leak_formulas, masked), "leak")
standard <- subset(carrier, condition == "standard")
out$H3_replication_secondary <- coefficient(fit_ladder(leak_formulas, standard), "leak")
contrast_with_standard <- function(name) {
  data <- subset(carrier, condition %in% c("standard", name))
  data$treated <- ifelse(data$condition == name, 0.5, -0.5)
  coefficient(fit_ladder(list(
    full = value ~ leak * treated + div + (1 + leak | architecture) + (1 | seed) + (1 | training_set),
    intercepts = value ~ leak * treated + div + (1 | architecture) + (1 | seed) + (1 | training_set)
  ), data), "leak:treated", sign = -1)
}
out$F4 <- contrast_with_standard("colour")
out$masking_reduction_secondary <- contrast_with_standard("masked")
out$carrier_cell_means <- aggregate(value ~ condition + cell + architecture, data = carrier, FUN = mean)

# ---------------- shared machinery for the dose-response arms ----------------
pair_runs <- function(frame, level) {
  keys <- c("architecture", "seed", level)
  merged <- merge(
    subset(frame, protocol == "P1")[, c(keys, "value", "R")],
    subset(frame, protocol == "P2")[, c(keys, "value", "R")],
    by = keys, suffixes = c("_p1", "_p2")
  )
  merged$gap <- merged$value_p1 - merged$value_p2
  merged$R_pair <- (merged$R_p1 + merged$R_p2) / 2
  merged
}

# Regime and manipulation check, identical to H5: levels whose mean P2 accuracy is
# within 1 pp of the baseline level, minus diverged levels; at the strongest such
# level mean R_pair must fall by at least 25% of its above-floor baseline value.
regime <- function(frame, pairs, level, baseline_level, excluded = NULL) {
  checks <- list()
  eligible <- list()
  means <- aggregate(as.formula(paste("cbind(value, R, floor) ~ architecture + protocol +", level)),
                     data = frame, FUN = mean)
  for (arch_name in sort(unique(as.character(pairs$architecture)))) {
    p2 <- means[means$architecture == arch_name & means$protocol == "P2", ]
    baseline <- p2$value[p2[[level]] == baseline_level]
    levels <- sort(p2[[level]][p2$value >= baseline - 1.0])
    if (!is.null(excluded)) levels <- setdiff(levels, excluded[[level]][excluded$architecture == arch_name])
    strongest <- if (baseline_level == min(p2[[level]])) max(levels) else min(levels)
    arch_pairs <- pairs[pairs$architecture == arch_name, ]
    floor_value <- mean(frame$floor[frame$architecture == arch_name])
    r0 <- mean(arch_pairs$R_pair[arch_pairs[[level]] == baseline_level])
    r_star <- mean(arch_pairs$R_pair[arch_pairs[[level]] == strongest])
    passed <- strongest != baseline_level && r_star <= r0 - 0.25 * (r0 - floor_value)
    checks[[arch_name]] <- list(levels_in_regime = levels, strongest = strongest, R_baseline = r0, R_strongest = r_star,
                                within_class_floor = floor_value, manipulation_check_passed = passed)
    if (passed) eligible[[arch_name]] <- arch_pairs[arch_pairs[[level]] %in% levels, ]
  }
  list(checks = checks, eligible = eligible, means = means)
}

# ---------------- F5: LEACE erasure ladder on Phase 1 checkpoints ----------------
leace <- read.csv(file.path(input, "leace_runs.csv"), stringsAsFactors = FALSE)
expect_rows(leace, 12 * 20 * 2 * 5, "LEACE table")
leace$value <- 100 * leace$accuracy
leace$R <- 100 * leace$within_class_accuracy
leace$floor <- 100 * leace$within_class_floor
leace_pairs <- pair_runs(leace, "rank")
leace_regime <- regime(leace, leace_pairs, "rank", 0)
out$F5_manipulation_checks <- leace_regime$checks
out$leace_means <- leace_regime$means
if (length(leace_regime$eligible) == 0) {
  out$F5 <- list(status = "untestable: erasure failed its manipulation check", p = NA, estimate = NA)
} else {
  reductions <- do.call(rbind, lapply(names(leace_regime$eligible), function(arch_name) {
    pairs <- leace_regime$eligible[[arch_name]]
    strongest <- leace_regime$checks[[arch_name]]$strongest
    base <- pairs[pairs$rank == 0, c("seed", "gap")]
    erased <- pairs[pairs$rank == strongest, c("seed", "gap")]
    merged <- merge(base, erased, by = "seed", suffixes = c("_0", "_star"))
    data.frame(architecture = arch_name, seed = merged$seed, reduction = merged$gap_0 - merged$gap_star)
  }))
  reductions$architecture <- factor(reductions$architecture)
  reductions$seed <- factor(reductions$seed)
  out$F5 <- coefficient(fit_ladder(list(
    crossed = reduction ~ 1 + (1 | architecture) + (1 | seed),
    architecture = reduction ~ 1 + (1 | architecture)
  ), reductions), "(Intercept)")
  out$F5$architectures <- levels(reductions$architecture)
  out$F5_reductions <- aggregate(reduction ~ architecture, data = reductions, FUN = mean)
}

# ---------------- F6: stain-normalisation dose ----------------
normal <- read.csv(file.path(input, "normalization_recoverability.csv"), stringsAsFactors = FALSE)
expect_rows(normal, 4 * 10 * 3 * 2, "normalisation table")
normal$value <- 100 * normal$accuracy
normal$R <- 100 * normal$within_class_accuracy
normal$floor <- 100 * normal$within_class_floor
normal_pairs <- pair_runs(normal, "strength")
normal_pairs$architecture <- factor(normal_pairs$architecture)
normal_pairs$seed <- factor(normal_pairs$seed)
out$F6 <- coefficient(fit_ladder(list(
  fixed_architecture = gap ~ architecture + strength + (1 | seed)
), normal_pairs), "strength", sign = -1)
normal_regime <- regime(normal, normal_pairs, "strength", 0)
out$F6_regime_secondary <- normal_regime$checks
out$normalization_means <- normal_regime$means

# ---------------- F7: probe-matched linear adversary ----------------
linear <- read.csv(file.path(input, "linadv_recoverability.csv"), stringsAsFactors = FALSE)
diverged <- read.csv(file.path(input, "linadv_diverged.csv"), stringsAsFactors = FALSE)
expect_rows(linear, 3 * 5 * 2 * 10 - nrow(diverged), "linear-adversary table")
linear$value <- 100 * linear$accuracy
linear$R <- 100 * linear$within_class_accuracy
linear$floor <- 100 * linear$within_class_floor
linear_pairs <- pair_runs(linear, "adversary_lambda")
linear_regime <- regime(linear, linear_pairs, "adversary_lambda", 0, excluded = diverged)
out$F7_manipulation_checks <- linear_regime$checks
out$linadv_means <- linear_regime$means
out$linadv_diverged <- diverged
if (length(linear_regime$eligible) == 0) {
  out$F7 <- list(status = "untestable: the linear adversary failed its manipulation check", p = NA, estimate = NA)
} else {
  data <- do.call(rbind, linear_regime$eligible)
  data$seed <- factor(data$seed)
  data$architecture <- factor(data$architecture)
  formula <- if (nlevels(data$architecture) > 1) gap ~ architecture + R_pair + (1 | seed) else gap ~ R_pair + (1 | seed)
  out$F7 <- coefficient(fit_ladder(list(fixed_architecture = formula), data), "R_pair")
  out$F7$architectures <- levels(data$architecture)
}

# ---------------- F8: pooled dose-response across manipulations ----------------
stack <- function(eligible, manipulation, level) {
  if (length(eligible) == 0) return(NULL)
  frame <- do.call(rbind, eligible)
  data.frame(manipulation = manipulation, architecture = as.character(frame$architecture),
             seed = as.character(frame$seed), level = frame[[level]], gap = frame$gap, R_pair = frame$R_pair)
}
pooled <- do.call(rbind, list(
  stack(leace_regime$eligible, "leace", "rank"),
  stack(normal_regime$eligible, "normalization", "strength"),
  stack(linear_regime$eligible, "linear_adversary", "adversary_lambda")
))
if (is.null(pooled) || nrow(pooled) == 0) {
  out$F8 <- list(status = "untestable: no manipulation passed its check", p = NA, estimate = NA)
} else {
  pooled$cell <- interaction(pooled$manipulation, pooled$architecture, drop = TRUE)
  pooled$block <- interaction(pooled$manipulation, pooled$seed, drop = TRUE)
  out$F8 <- coefficient(fit_ladder(list(
    within_cell = gap ~ cell + R_pair + (1 | block)
  ), pooled), "R_pair")
  out$F8$manipulations <- sort(unique(pooled$manipulation))
  out$F8$cells <- nlevels(pooled$cell)
}

# ---------------- Holm family F1-F8 ----------------
cric <- fromJSON(args[[2]])
out$F1 <- list(estimate = cric$F1$estimate, p = cric$F1$p_two_sided, ci95 = cric$F1$ci95, source = "slide bootstrap")
names_f <- paste0("F", 1:8)
p_values <- sapply(names_f, function(h) if (is.null(out[[h]]$p)) NA_real_ else as.numeric(out[[h]]$p))
estimates <- sapply(names_f, function(h) if (is.null(out[[h]]$estimate)) NA_real_ else as.numeric(out[[h]]$estimate))
testable <- !is.na(p_values)
adjusted <- rep(NA_real_, length(p_values))
names(adjusted) <- names_f
adjusted[testable] <- p.adjust(p_values[testable], method = "holm")
out$holm <- lapply(names_f, function(h) list(
  hypothesis = h, p = unname(p_values[h]), p_holm = unname(adjusted[h]), estimate = unname(estimates[h]),
  supported = isTRUE(adjusted[h] < 0.05 && estimates[h] > 0)
))
write_json(out, args[[3]], auto_unbox = TRUE, pretty = TRUE, digits = NA)
for (item in out$holm) {
  cat(sprintf("%s estimate=%.3f p=%.4g p_holm=%.4g supported=%s\n",
              item$hypothesis, item$estimate, item$p, item$p_holm, item$supported))
}
