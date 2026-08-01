library(readr)
library(dplyr)
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
results_dir <- file.path(project_root, "results")
out_dir <- file.path(project_root, "R")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

group_levels <- c(
  "Self protective behaviours",
  "Demographics",
  "Health, mental health and wellbeing",
  "Perception of illness threat",
  "Time",
  "Other"
)

colour_palette <- c(
  "Self protective behaviours" = "#2FB344",
  "Demographics" = "#5B8DEF",
  "Health, mental health and wellbeing" = "#F59F00",
  "Perception of illness threat" = "#D6336C",
  "Time" = "#7950F2",
  "Other" = "#868E96"
)

feature_group <- function(name) {
  case_when(
    str_detect(name, "i2|i9|i11|protective") ~ "Self protective behaviours",
    str_detect(name, "age|house|employ|gender|state") ~ "Demographics",
    str_detect(name, "PHQ|cantril|d1") ~ "Health, mental health and wellbeing",
    str_detect(name, "r1") ~ "Perception of illness threat",
    str_detect(name, "week") ~ "Time",
    TRUE ~ "Other"
  )
}

clean_label <- function(labels_original) {
  labels <- labels_original |>
    str_replace_all("_", " ") |>
    str_to_title()

  labels[str_detect(labels, "I2 Health")] <- "Non-Household Contacts"
  labels[str_detect(labels, "R1 1")] <- "Perceived Severity"
  labels[str_detect(labels, "R1 2")] <- "Perceived Susceptibility"
  labels[str_detect(labels, "Cantril Ladder")] <- "Life Satisfaction"
  labels[str_detect(labels, "Week Number")] <- "Two-Week Period"
  labels[str_detect(labels, "Household Size")] <- "Household Size"

  labels[str_detect(labels, "I9 Health")] <- labels[str_detect(labels, "I9 Health")] |>
    str_replace("I9 Health ", "Isolate If Unwell (") |>
    str_c(")")

  labels[str_detect(labels, "I11 Health")] <- labels[str_detect(labels, "I11 Health")] |>
    str_replace("I11 Health ", "") |>
    str_c(" To Isolate")

  labels[str_detect(labels, "State")] <- labels[str_detect(labels, "State")] |>
    str_replace("State ", "State (") |>
    str_c(")")

  labels
}

load_top10 <- function(model_type, task, period) {
  read_csv(
    file.path(results_dir, paste0(task, "_", model_type, "_top10_features.csv")),
    show_col_types = FALSE
  ) |>
    mutate(
      model_type = model_type,
      period = period,
      group = factor(feature_group(feature), levels = group_levels),
      feature_label = clean_label(feature)
    )
}

make_model_data <- function(model_type) {
  bind_rows(
    load_top10(model_type, "model_3a", "Before face mask mandate"),
    load_top10(model_type, "model_3b", "After face mask mandate")
  ) |>
    group_by(period) |>
    arrange(importance_mean, .by_group = TRUE) |>
    mutate(
      feature_period = paste(feature_label, period, sep = "___"),
      feature_period = factor(feature_period, levels = feature_period)
    ) |>
    ungroup() |>
    mutate(period = factor(period, levels = c("Before face mask mandate", "After face mask mandate")))
}

plot_top10 <- function(df, title_text) {
  ggplot(df, aes(x = importance_mean, y = feature_period, fill = group)) +
    geom_col(width = 0.72) +
    geom_errorbarh(
      aes(xmin = pmax(importance_mean - importance_std, 0), xmax = importance_mean + importance_std),
      height = 0.25,
      linewidth = 0.45,
      colour = "#212529"
    ) +
    facet_wrap(~period, scales = "free_y") +
    scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
    scale_fill_manual(values = colour_palette, drop = TRUE) +
    labs(title = title_text, x = "Mean feature importance", y = NULL, fill = NULL) +
    theme_bw(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 18),
      strip.text = element_text(face = "bold", size = 12),
      axis.text.y = element_text(size = 9),
      axis.title.x = element_text(margin = margin(t = 8)),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 9),
      plot.margin = margin(10, 12, 10, 10)
    ) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE))
}

rf_data <- make_model_data("rf")
xgb_data <- make_model_data("xgboost")

write_csv(rf_data, file.path(out_dir, "social_distance_rf_top10_feature_plot_data.csv"))
write_csv(xgb_data, file.path(out_dir, "social_distance_xgboost_top10_feature_plot_data.csv"))

p_rf <- plot_top10(rf_data, "Random Forest Top 10 Feature Importance")
p_xgb <- plot_top10(xgb_data, "XGBoost Top 10 Feature Importance")

ggsave(
  file.path(out_dir, "social_distance_rf_top10_feature_importance.png"),
  p_rf,
  width = 12,
  height = 7.8,
  dpi = 300
)

ggsave(
  file.path(out_dir, "social_distance_xgboost_top10_feature_importance.png"),
  p_xgb,
  width = 12,
  height = 7.8,
  dpi = 300
)

combined <- (p_rf + theme(legend.position = "none")) + p_xgb +
  plot_layout(widths = c(1, 1)) &
  theme(legend.position = "bottom")

ggsave(
  file.path(out_dir, "social_distance_rf_xgboost_top10_feature_importance.png"),
  combined,
  width = 18,
  height = 8.2,
  dpi = 300
)
