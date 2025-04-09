process MUTRAIN {
    label 'process_high'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/mutrain", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(modelfoler)

    output:
    val(meta), emit: meta_id
    path("train/*"), emit: train_dirs
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}_train"
    def folders = modelfoler instanceof List ? modelfoler.join(',') : [modelfoler].join(',')

    """
    mutrainco.py --folder ${folders}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        torch: \$(python -c "import torch; print(torch.__version__,torch.cuda.is_available())")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}_train"
    """
    touch "${prefix}_train.pdf"
    touch "versions.yml"
    """


}

