process TRAIN {
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/train", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    val(json1)
    path(npz , stageAs : "res.npz")

    output:
    tuple val(meta), val(reads)
    path("res.npz")

    script:
    """
    train.py \
    --meta ${meta} \
    --reads ${reads} \
    --json ${json1} \
    --npz ${npz} \
    """



}

