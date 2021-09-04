select distinct dx.dx_name as code,
                dx.timestamp_utc as order_id, -- proxy because dx table doesn't have order_id 
                'Diagnosis' as feature_type,
                cohort.anon_id, cohort.pat_enc_csn_id_coded, cohort.admit_time, 
                cohort.first_label, cohort.death_24hr_recent_label
from
  triageTD.1_5_cohort_final as cohort as cohort,
  shc_core.diagnosis_code as dx
where
  cohort.anon_id = dx.anon_id
  and cohort.pat_enc_csn_id_coded <> dx.pat_enc_csn_id_coded
  and timestamp_add(dx.timestamp_utc, INTERVAL 1 HOUR) < cohort.admit_time 
