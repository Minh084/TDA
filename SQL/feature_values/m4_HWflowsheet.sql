SELECT c.*,
    f.row_disp_name, f.units, f.recorded_time_utc, f.num_value1, f.num_value2
FROM 
    `som-nero-phi-jonc101.triageTD.cohort_enc_code_age` as c -- # old: traige_TE.triage_cohort_adjusted
JOIN 
    `som-nero-phi-jonc101.shc_core.flowsheet` as f
ON 
    (c.anon_id=f.anon_id and c.inpatient_data_id_coded=f.inpatient_data_id_coded)
WHERE
(
(row_disp_name="Weight")
OR
(row_disp_name="Height")
)
