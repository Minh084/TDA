SELECT c.anon_id, c.pat_enc_csn_id_coded, c.inpatient_data_id_coded,
    f.row_disp_name, f.units, f.recorded_time_utc, f.num_value1, f.num_value2
FROM 
    `som-nero-phi-jonc101.triageTD.1_2_cohort` as c
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
