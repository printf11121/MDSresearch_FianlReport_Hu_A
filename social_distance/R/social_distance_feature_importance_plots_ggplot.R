library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(patchwork)

find_project_root <- function() {
  candidates <- normalizePath(c(".", ".."), mustWork = FALSE)
  roots <- candidates[
    dir.exists(file.path(candidates, "results")) &
      dir.exists(file.path(candidates, "data"))
  ]
  if (length(roots) == 0) {
    stop("Could not locate the social_distance project root.")
  }
  roots[[1]]
}

project_root <- find_project_root()
out_dir <- file.path(project_root, "R")
results_dir <- file.path(project_root, "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

group_levels <- c(
  "Self protective behaviours",
  "Demographics",
  "Health, mental health and wellbeing",
  "Perception of illness threat",
  "Time",
  "Trust in government",
  "Other"
)

colour_palette <- c(
  "Self protective behaviours" = "#2FB344",
  "Demographics" = "#5B8DEF",
  "Health, mental health and wellbeing" = "#F59F00",
  "Perception of illness threat" = "#D6336C",
  "Time" = "#7950F2",
  "Trust in government" = "#0CA678",
  "Other" = "#868E96"
)

feature_group <- function(name) {
  case_when(
    str_detect(name, "i2|i9|i11|protective") ~ "Self protective behaviours",
    str_detect(name, "age|house|employ|gender|state") ~ "Demographics",
    str_detect(name, "PHQ|cantril|d1") ~ "Health, mental health and wellbeing",
    str_detect(name, "r1") ~ "Perception of illness threat",
    str_detect(name, "week") ~ "Time",
    str_detect(name, "WCR") ~ "Trust in government",
    TRUE ~ "Other"
  )
}

clean_label <- function(labels_original) {
  labels <- labels_original %>%
    str_replace_all("_", " ") %>%
    str_to_title()

  labels[str_detect(labels, "I2 Health")] <- "Non-Household Contacts"
  labels[str_detect(labels, "R1 1")] <- "Perceived Severity"
  labels[str_detect(labels, "R1 2")] <- "Perceived Susceptibility"
  labels[str_detect(labels, "Cantril Ladder")] <- "Life Satisfaction"
  labels[str_detect(labels, "Household Size")] <- "Household Size"
  labels[str_detect(labels, "Week Number")] <- "Two-Week Period"
  labels[str_detect(labels, "Gender Male")] <- "Gender (Male)"

  labels[str_detect(labels, "I9 Health")] <- labels[str_detect(labels, "I9 Health")] %>%
    str_replace("I9 Health ", "Isolate If Unwell (") %>%
    str_c(")")

  labels[str_detect(labels, "I11 Health")] <- labels[str_detect(labels, "I11 Health")] %>%
    str_replace("I11 Health ", "") %>%
    str_c(" To Isolate")

  labels[str_detect(labels, "Employment Status")] <- labels[str_detect(labels, "Employment Status")] %>%
    str_replace("Employment Status ", "Employment Status (") %>%
    str_c(")")

  labels[str_detect(labels, "Phq4 1")] <- labels[str_detect(labels, "Phq4 1")] %>%
    str_replace("Phq4 1 ", "Little Interest Or Pleasure (") %>%
    str_c(")")
  labels[str_detect(labels, "Phq4 2")] <- labels[str_detect(labels, "Phq4 2")] %>%
    str_replace("Phq4 2 ", "Feeling Down Or Depressed (") %>%
    str_c(")")
  labels[str_detect(labels, "Phq4 3")] <- labels[str_detect(labels, "Phq4 3")] %>%
    str_replace("Phq4 3 ", "Feeling Nervous Or Anxious (") %>%
    str_c(")")
  labels[str_detect(labels, "Phq4 4")] <- labels[str_detect(labels, "Phq4 4")] %>%
    str_replace("Phq4 4 ", "Worrying (") %>%
    str_c(")")

  labels[str_detect(labels, "Wcrex2")] <- labels[str_detect(labels, "Wcrex2")] %>%
    str_replace("Wcrex2 ", "Confidence In Response (") %>%
    str_c(")")

  labels[str_detect(labels, "D1 Comorbidities")] <- labels[str_detect(labels, "D1 Comorbidities")] %>%
    str_replace("D1 Comorbidities ", "Has Comorbidities (") %>%
    str_c(")")

  labels
}

summarise_importance <- function(path, period) {
  read_csv(path, show_col_types = FALSE) %>%
    pivot_longer(everything(), names_to = "feature", values_to = "importance") %>%
    group_by(feature) %>%
    summarise(
      median_importance = median(importance, na.rm = TRUE),
      q1 = quantile(importance, 0.25, na.rm = TRUE),
      q3 = quantile(importance, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(period = period, group = feature_group(feature))
}

make_plot_data <- function(model_type) {
  before <- summarise_importance(
    file.path(results_dir, paste0("model_3a_", model_type, "_feature_importance.csv")),
    "Before face mask mandate"
  )
  after <- summarise_importance(
    file.path(results_dir, paste0("model_3b_", model_type, "_feature_importance.csv")),
    "After face mask mandate"
  )

  all <- bind_rows(before, after) %>%
    filter(!str_detect(feature, "state"))

  top_before <- before %>%
    filter(!str_detect(feature, "state")) %>%
    slice_max(median_importance, n = 10, with_ties = FALSE) %>%
    pull(feature)
  top_after <- after %>%
    filter(!str_detect(feature, "state")) %>%
    slice_max(median_importance, n = 10, with_ties = FALSE) %>%
    pull(feature)

  selected <- union(top_before, top_after)

  all %>%
    filter(feature %in% selected) %>%
    group_by(feature) %>%
    mutate(total_importance = sum(median_importance)) %>%
    ungroup() %>%
    mutate(
      feature = fct_reorder(feature, total_importance),
      signed_importance = if_else(period == "Before face mask mandate", -median_importance, median_importance),
      signed_q1 = if_else(period == "Before face mask mandate", -q1, q1),
      signed_q3 = if_else(period == "Before face mask mandate", -q3, q3),
      group = factor(group, levels = group_levels),
      period = factor(period, levels = c("Before face mask mandate", "After face mask mandate"))
    )
}

make_tornado_plot <- function(plot_data, title) {
  labels <- clean_label(levels(plot_data$feature))
  names(labels) <- levels(plot_data$feature)
  max_x <- max(abs(c(plot_data$signed_importance, plot_data$signed_q1, plot_data$signed_q3)), na.rm = TRUE)

  ggplot(plot_data, aes(x = signed_importance, y = feature, fill = group)) +
    geom_col(width = 0.62, show.legend = TRUE) +
    geom_errorbarh(aes(xmin = pmin(signed_q1, signed_q3), xmax = pmax(signed_q1, signed_q3)),
                   height = 0.24, linewidth = 0.45, colour = "#212529") +
    geom_vline(xintercept = 0, colour = "#343A40", linewidth = 0.45) +
    annotate("text", x = -max_x * 0.38, y = 0.2, label = "Before face mask mandate", size = 3.5) +
    annotate("text", x = max_x * 0.38, y = 0.2, label = "After face mask mandate", size = 3.5) +
    scale_x_continuous(labels = function(x) signif(abs(x), 3), expand = expansion(mult = c(0.03, 0.03))) +
    scale_y_discrete(labels = labels, expand = expansion(add = c(1.0, 0.6))) +
    scale_fill_manual(values = colour_palette, drop = FALSE) +
    coord_cartesian(clip = "off") +
    labs(title = title, x = "Median feature importance", y = NULL, fill = NULL) +
    theme_bw(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
      axis.text.y = element_text(size = 10),
      axis.title.x = element_text(size = 13, margin = margin(t = 10)),
      legend.position = "bottom",
      legend.text = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 18, 34, 10)
    ) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE))
}

rf_data <- make_plot_data("rf")
xgb_data <- make_plot_data("xgboost")

write_csv(rf_data, file.path(out_dir, "social_distance_rf_feature_importance_plot_data_ggplot.csv"))
write_csv(xgb_data, file.path(out_dir, "social_distance_xgboost_feature_importance_plot_data_ggplot.csv"))

p_rf <- make_tornado_plot(rf_data, "Random Forest Feature Importance")
p_xgb <- make_tornado_plot(xgb_data, "XGBoost Feature Importance")

ggsave(file.path(out_dir, "social_distance_rf_feature_importance_tornado_ggplot.png"),
       p_rf, width = 11, height = 8.2, dpi = 300)
ggsave(file.path(out_dir, "social_distance_xgboost_feature_importance_tornado_ggplot.png"),
       p_xgb, width = 11, height = 8.2, dpi = 300)

combined <- (p_rf + p_xgb) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "social_distance_rf_xgboost_feature_importance_tornado_ggplot.png"),
       combined, width = 17, height = 8.2, dpi = 300)
