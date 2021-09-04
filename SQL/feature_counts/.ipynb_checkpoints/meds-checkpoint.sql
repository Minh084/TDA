select distinct med.order_med_id_coded,
                med.med_description as code,
                med.order_med_id_coded as order_id,
                'Meds' as feature_type,
--                 med.pharm_class_name,
--                 med.thera_class_name,
--                 med.order_class,
                med.order_time_jittered_utc as med_order_time,
                cohort.anon_id, cohort.pat_enc_csn_id_coded, cohort.admit_time, 
                cohort.first_label, cohort.death_24hr_recent_label
from
  triageTD.1_5_cohort_final as cohort,
  shc_core.order_med as med
where
  cohort.anon_id = med.anon_id
  and timestamp_add(med.order_time_jittered_utc, INTERVAL 1 HOUR)< cohort.admit_time
  and timestamp_add(med.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= cohort.admit_time