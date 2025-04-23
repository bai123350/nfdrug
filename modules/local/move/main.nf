process MOVE {
    tag "$meta.id"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://kjd12/pyhtonbio:1.0' :
        'kjd12/pyhtonbio:1.0' }"

    publishDir "${params.outdir}/move", mode: 'copy'

    input:
    tuple val(meta), val(reads)
    path(folders)


    output:
    val(meta),  emit: meta_id
    path("csv/*"), emit: csv
    path("pdf/*"), emit: pdf

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "move"
    """
    mkdir -p pdf
    mkdir -p csv
    for item in ${folders}; do
        if [[ -d \${item} ]]; then
            for file in \${item}/*; do
                if [[ \${file} == *.pdf ]]; then
                    cp \${file} pdf/
                fi
            done
        elif [[ \${item} == *.pdf ]]; then
            cp \${item} pdf/
        elif [[ \${item} == *.csv ]]; then
            cp \${item} csv/
        else
            echo "ERROR: \${item} is not a directory or a pdf file"
        fi
    done
    """

    stub:
    def prefix = task.ext.prefix ?: "move"
    """
    touch "${prefix}_empty.csv"
    touch "${prefix}_empty.pdf"
    """
}
