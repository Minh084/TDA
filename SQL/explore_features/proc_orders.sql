select distinct lab.order_proc_id_coded as order_id,
                CASE WHEN lab.order_type in('Lab', 'Microbiology', 'Microbiology Culture') THEN 'Lab' ELSE lab.order_type END as feature_type,
                lab.proc_code as code,
                lab.order_time_jittered_utc,
                cohort.anon_id, cohort.pat_enc_csn_id_coded, cohort.admit_time, 
                cohort.first_label, cohort.death_24hr_recent_label
from
  triageTD.1_5_cohort_final as cohort,
  shc_core.order_proc as lab
where
  lab.order_type in ('Lab', 'Microbiology', 'Microbiology Culture', 'Imaging', 'Procedures')
  and cohort.anon_id = lab.anon_id
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 1 HOUR) < cohort.admit_time
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= cohort.admit_time
