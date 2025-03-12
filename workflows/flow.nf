


// include {} from '../modules/nf-core/models/main'

include { DATATETCH               } from '../modules/nf-core/datafetch/main'



workflow RUNDATA {
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    // WORKFLOW: Run pipeline
    // MOLFLOW (
    //     samplesheet
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // MODULE: Run FastQC
    DATATETCH (
        samplesheet
    )


    emit:
    multiqc_report = DATATETCH.out.html // channel: /path/to/multiqc_report.html
}
