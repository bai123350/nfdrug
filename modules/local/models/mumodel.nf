process MUALLMODELS {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/mumodels", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(all_folders)

    output:
    tuple val(meta), emit: meta_id
    path("*.pt"), emit: pt
    path("*.npz"), emit: npz
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_models"
    def xy = task.ext.xy ?: "${meta.id}_xy"
    // def js = json1.find { it.toString().contains('basic') }
    def dis = task.ext.dis ?: "${params.disea}"

    """


    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}_models"
    """
    touch "${prefix}_models.npz"
    touch "versions.yml"
    """

}
