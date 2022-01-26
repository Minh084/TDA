library(ggplot2)
library(tidyverse)

# plot the positive/negative labels by time in ED


maindir <- "D:/school/triage/"
datadir <- "OutputTD/5_results_analysis/"

# read in the labels across different time points
labels_file <- paste0(maindir, datadir, "5_4_time_range_labels.csv")
labels <- read_csv(labels_file, col_select = -1)
head(labels)
dim(labels)

# get total number of csns
csns_length <- length(unique(labels$pat_enc_csn_id_coded))
csns_length

# select label columns and make long
cleaned_labels <- labels %>%
  select(anon_id, pat_enc_csn_id_coded, contains('label_'))

# count the number of transfers for each hour
hours <- 1:47
hour <- 1
count_transfers <- function(hour){
  head(cleaned_labels)
  curr_label <- paste0("label_", hour, 'hr')
  last_label <- paste0('label_', hour-1, 'hr')
  curr_label
  last_label
  
  # count positive transfers
  # curr_label == 1 and last_label == 0
  pos_count <- sum(cleaned_labels[curr_label] - cleaned_labels[last_label] == 1)
  pos_count
  
  # count negative transfers
  # curr_label == 0 and last_label == 1
  neg_count <- sum(cleaned_labels[curr_label] - cleaned_labels[last_label] == -1)
  neg_count
  
  data.frame(hour=hour, positive=pos_count, negative=neg_count)
}

all_counts <- lapply(hours, count_transfers)
counts_df <- do.call(rbind, all_counts)
counts_df


cleaned_labels <- labels %>%
  select(anon_id, pat_enc_csn_id_coded, contains('label_')) %>%
  filter(label_0hr == 0) %>%
  gather('hour', 'label', -anon_id, -pat_enc_csn_id_coded) %>%
  filter(hour != 'label_0hr') %>%
  mutate(hour=str_remove(hour, 'label_')) %>%
  mutate(hour=str_remove(hour, "hr"))
head(cleaned_labels)



# count the types of labels per hour
label_counts <- cleaned_labels %>%
  filter(label==1) %>%
  group_by(hour) %>%
  summarise(positive=sum(label)) %>%
  mutate(negative=csns_length - positive)
  # gather('label_type', 'count', -hour)
head(label_counts)

# adjust hour factor
unique(label_counts$hour)
label_counts$hour <- factor(label_counts$hour,
                            levels=1:47)
# label_counts$label_type <- factor(label_counts$label_type,
#                                   levels=c('positive', 'negative'))


# options(repr.plot.width=8, repr.plot.height=7)

head(label_counts)

colors <- c('negative'='red', 'positive'='dodgerblue1')

ggplot(label_counts) +
  geom_bar(aes(x=hour, y=negative, fill='negative',), 
                 stat='identity', position='identity',
           alpha=0.7) +
  geom_bar(aes(x=hour, y=positive, fill='positive',), 
           stat='identity', position='identity',
           alpha=0.7) +
  labs(x='Hours since initial inpatient admission', 
       y='Label Count') +
  scale_fill_manual(values=colors)
  
  

ggplot(pred2, aes(x = predictions, fill = labels)) + 
  geom_histogram(position = "identity", alpha = 0.2, bins = 50) +
  labs(title = "Overlapping Histograms of Predicted Risks\n", 
       x = "Predicted probability", y = "Frequency") +
  scale_fill_discrete(name = "Prediction times\n", labels = c("t0", "t24")) +
  #     scale_color_manual(labels = c("Initial Admission", "24th hour after"), values = c("blue", "red")) +
  theme(axis.text = element_text(size=14),
        plot.title = element_text(size=16, hjust=0.5),
        axis.title = element_text(size=16), legend.key.size = unit(1, "cm"), 
        legend.text=element_text(size=13), legend.title=element_text(size=13))#, face="bold"

ggsave(filename = file.path(figuredir, "Fig3.1_histograms024.png"), width = 8, height = 7, dpi = 1200)