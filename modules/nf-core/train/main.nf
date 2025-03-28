process TRAIN {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/train", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    val(pt)
    path(npz , stageAs : "xylabel.npz")

    output:
    tuple val(meta), emit: meta_id
    path("*.pdf"), emit: pdf
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_train"
    def traindata = pt.find { it.toString().contains('train') }
    def model = pt.find { it.toString().contains('model') }
    """
    train.py \
    --model ${model} \
    --train ${traindata} \
    --m ${prefix} 

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        torch: \$(python -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}_train"
    """
    touch "${prefix}_train.pdf"
    touch "versions.yml"
    """


}

