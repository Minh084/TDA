# Sequence of codes assigned to patients 

WITH sequence as (
# Get (lab, procedure, imaging and microbiology orders from order_proc) from last year
SELECT DISTINCT lab.order_proc_id_coded as order_id,
lab.order_type as feature_type,
lab.description as features,
labels.*
FROM traige_TE.triage_cohort_final_with_labels as labels
INNER JOIN shc_core.order_proc as lab
USING (anon_id)
WHERE 
(lab.order_type in ('Imaging', 'Procedures', 'Lab')
AND lab.order_time_jittered_utc < labels.admit_time
AND timestamp_add(lab.order_time_jittered_utc, INTERVAL 24*365 HOUR) >= labels.admit_time)
OR
(lab.order_type in ('Microbiology', 'Microbiology Culture')
AND lab.order_time_jittered_utc < labels.admit_time
AND lab.pat_enc_csn_id_coded = labels.pat_enc_csn_id_coded)
UNION DISTINCT

# Get icd10 codes from full timeline
SELECT DISTINCT
pat_enc_csn_id_coded as order_id, -- fill in because dx table doesn't have order_id
'Diagnosis' as feature_type,
dx.icd10 as features,
labels.*
FROM
traige_TE.triage_cohort_final_with_labels as labels
INNER JOIN shc_core.diagnosis_code as dx
USING (anon_id)
WHERE labels.pat_enc_csn_id_coded <> dx.pat_enc_csn_id_jittered
AND dx.start_date_utc < labels.admit_time 
AND dx.icd10 is not NULL
UNION DISTINCT

# Get med orders from order_med from last year
SELECT DISTINCT
med.order_med_id_coded order_id,
'Meds' as feature_type,
med.med_description as features,
labels.*
FROM traige_TE.triage_cohort_final_with_labels as labels
INNER JOIN shc_core.order_med as med
USING (anon_id)
WHERE med.order_start_time_utc < labels.admit_time
AND timestamp_add(med.order_start_time_utc, INTERVAL 24*365 HOUR) >= labels.admit_time
AND med.order_start_time_utc IS NOT NULL) -- maybe redundent but fine

# Group by feature and encounter, and count # distinct orders/assignemnts of the feature
SELECT anon_id, pat_enc_csn_id_coded, admit_time, feature_type, features, COUNT (DISTINCT order_id) as values
FROM sequence
GROUP BY anon_id, pat_enc_csn_id_coded, admit_time, feature_type, features
ORDER BY pat_enc_csn_id_coded