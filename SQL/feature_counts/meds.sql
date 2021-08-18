select distinct med.order_med_id_coded,
                med.med_description as code,
                med.order_med_id_coded as order_id,
                'Meds' as feature_type,
--                 med.pharm_class_name,
--                 med.thera_class_name,
--                 med.order_class,
                med.order_time_jittered_utc as med_order_time,
                labels.*
from
  traige_TE.triage_cohort_adjusted as labels,
  starr_datalake2018.order_med as med
where
  labels.jc_uid = med.jc_uid
  and timestamp_add(med.order_time_jittered_utc, INTERVAL 1 HOUR)< labels.admit_time
  and timestamp_add(med.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= labels.admit_time