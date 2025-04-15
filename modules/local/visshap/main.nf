process VISSHAP {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visshap", mode: 'copy'

    input:
    tuple val(sample_id), path(input_file)

    output:
    path("*.shap"), emit: shap_files


    script:
    """
    shap.py --input ${input_file} --output ${sample_id}.shap
    """
}
