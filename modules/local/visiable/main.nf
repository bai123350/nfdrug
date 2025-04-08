process VISIABLE {

    label 'process_high_memory'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visiable", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(modelfoler)

    output:
    path("") , emit : pdf
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:

    def prefix = task.ext.prefix ?: "_visable"
    """

    """

    stub:
    """

    """
}

