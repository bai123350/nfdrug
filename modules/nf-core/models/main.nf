
process MODELS {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/dataprocess", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(json)

    output:
    tuple val(meta), path("${meta.sample}.models.tsv")

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_models"

    """

    """

    stub:
    """
    touch "models.tsv"
    """

}