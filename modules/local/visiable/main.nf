process VISIABLE {

    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    publishDir "${params.outdir}/visiable", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(trainfoler)

    output:
    // path("*pdf") , emit : pdf
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:

    def prefix = task.ext.prefix ?: "visable"
    def folders = trainfoler instanceof List ? trainfoler.join(',') : [trainfoler].join(',')
    """
    visiable.py --dir ${folders}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        numpy: \$(python -c "import numpy; print(numpy.__version__)")
        torch: \$(python -c "import torch; print(torch.__version__,torch.cuda.is_available())")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "visable"
    """
    touch "${prefix}_visable.pdf"
    touch "versions.yml"
    """
}

