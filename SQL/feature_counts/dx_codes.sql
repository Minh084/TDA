select distinct dx.dx_name as code,
                dx.timestamp_utc as order_id, -- proxy because dx table doesn't have order_id 
                'Diagnosis' as feature_type,
                labels.*
from
  traige_TE.triage_cohort_adjusted as labels,
  starr_datalake2018.diagnosis_code as dx
where
  labels.jc_uid = dx.jc_uid
  and labels.pat_enc_csn_id_coded <> dx.pat_enc_csn_id_coded
  and timestamp_add(dx.timestamp_utc, INTERVAL 1 HOUR) < labels.admit_time 
