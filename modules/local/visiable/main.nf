process VISIABLE {

    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visiable", mode: 'copy'

    input:
    val identifier

    output:
    val identifier

    script:
    """

    """

    stub:
    """

    """
}

