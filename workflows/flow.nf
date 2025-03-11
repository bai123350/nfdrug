


// include {} from '../modules/nf-core/models/main'

include { FASTQC                 } from '../modules/nf-core/fastqc/main'



workflow RUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    script:
    // WORKFLOW: Run pipeline
    // MOLFLOW (
    //     samplesheet
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run FastQC
    FASTQC (
        ch_samplesheet
    )
    // )

    emit:
    multiqc_report = FASTQC.out.multiqc_report // channel: /path/to/multiqc_report.html
}
