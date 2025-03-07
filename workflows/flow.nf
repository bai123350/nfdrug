


// include {} from '../modules/nf-core/models/main'





workflow  RUN_DATA{
    take:
    samplesheet // channel: samplesheet read in from --input

    main:
    //
    // WORKFLOW: Run pipeline
    //
    // MOLFLOW (
    //     samplesheet
    // )


    emit:
    multiqc_report = MOLFLOW.out.multiqc_report // channel: /path/to/multiqc_report.html






}
