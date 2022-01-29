library(ggplot2)
library(tidyverse)


datafile <- paste0("D:/school/triage/OutputTD/5_results_analysis/5_4_pos_neg_switch_count_per_hour.csv")

counts <- read_csv(datafile)
head(counts)

names(counts)[1] <- 'label'

# format the data
label_counts <- counts %>%
  mutate(hour = str_remove(label, "label_switch_")) %>%
  mutate(hour = str_remove(hour, "hr")) %>%
  select(-label) %>%
  gather('label_type', 'count', -hour)
head(label_counts)

label_counts$hour <- factor(label_counts$hour,
                            levels=1:24)



colors <- c('negative'='firebrick1', 'positive'='dodgerblue1')

ggplot(label_counts) +
  geom_bar(aes(x=hour, y=count, fill=label_type), 
           stat='identity', position='identity',
           alpha=0.7) +
  labs(
    # title = "Overlapping Histograms of Transfers per Hour\n", 
       x = "Hours since initial inpatient admission", 
       y = "Transfer counts") +
  scale_fill_manual(values=colors, labels = c("Negative (1 -> 0)", "Positive (0 -> 1)")) +
   theme(axis.text = element_text(size=14),
        # plot.title = element_text(size=16, hjust=0.5),
        axis.title = element_text(size=16), legend.key.size = unit(1, "cm"),
        legend.position="none") +
        # legend.text=element_text(size=13), legend.title=element_text(size=13)) + #, face="bold" 
  guides(fill=guide_legend(title="Care Level Switch"))

savefile <- paste0("D:/school/triage/OutputTD/5_results_analysis/5_4_pos_neg_switch_count_per_hour.png")
ggsave(savefile, width=9, height=6)

