select distinct c.anon_id, c.pat_enc_csn_id_coded, 
                dx.line, dx.dx_name, dx.primary, dx.chronic, dx.principal, dx.hospital_pl, dx.ed, dx.present_on_adm
from
  `som-nero-phi-jonc101.triageTD.1_4_cohort` as c
join
  `som-nero-phi-jonc101.shc_core.diagnosis_code` as dx
ON 
    (c.anon_id = dx.anon_id and c.pat_enc_csn_id_coded = dx.pat_enc_csn_id_jittered)