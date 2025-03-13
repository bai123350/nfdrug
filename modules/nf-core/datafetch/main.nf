
process DATATETCH {
    tag "$meta.id"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'https://depot.galaxyproject.org/singularity/fastqc:0.12.1--hdfd78af_0' :
    //     'biocontainers/fastqc:0.12.1--hdfd78af_0' }"
    publishDir "${params.outdir}/datafetch", mode: 'copy'
    // modules\local\example\resources\usr\bin
    input:
    tuple val(meta), val(reads)

    output:
    tuple val(meta), path("*.json"), emit: html
    // tuple val(meta), path("*.zip") , emit: zip
    path  "versions.yml"           , emit: versions

    script:
    """
    fetch.py --score ${params.score} --out "res.json" --path1 ${reads[0]} --path2 ${reads[1]} 
    """

    stub:
    """
    touch res.json
    touch versions.yml
    
    """
}






