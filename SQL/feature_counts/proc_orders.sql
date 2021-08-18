select distinct lab.order_proc_id_coded as order_id,
                CASE WHEN lab.order_type in('Lab', 'Microbiology', 'Microbiology Culture') THEN 'Lab' ELSE lab.order_type END as feature_type,
                lab.proc_code as code,
                lab.order_time_jittered_utc,
                labels.*
from
  traige_TE.triage_cohort_adjusted as labels,
  starr_datalake2018.order_proc as lab
where
  lab.order_type in ('Lab', 'Microbiology', 'Microbiology Culture', 'Imaging', 'Procedures')
  and labels.jc_uid = lab.jc_uid
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 1 HOUR) < labels.admit_time
  and timestamp_add(lab.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= labels.admit_time
