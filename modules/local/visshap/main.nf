process VISSHAP {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visshap", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(trainfoler)

    output:
    // path("*pdf") , emit : pdf
    path "versions.yml", emit: versions

    script:
    """
    shap.py
    """

}
