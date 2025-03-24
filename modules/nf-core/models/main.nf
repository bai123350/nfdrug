
process MODELS {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/models", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    val(json1)
    path(npz)

    output:
    tuple val(meta), emit: meta_id
    path("*.npz"), emit: npz
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_models"
    def xy = task.ext.xy ?: "${meta.id}_xy"

    """
    model.py --path1 ${reads[3]}  --path2 ${json1[0]} --group ${reads[4]} --disea ${params.disea}  --out  ${xy}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
    END_VERSIONS
    """

    stub:
    """
    touch "models.npz"
    touch "versions.yml"
    """

}