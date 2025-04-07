process VISIABLE {

    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visiable", mode: 'copy'

    input:
    val identifier

    output:
    path("") , emit : pdf
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:

    def prefix = task.ext.prefix ?: "${identifier}_visable"
    """

    """

    stub:
    """

    """
}

