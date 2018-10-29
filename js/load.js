function load_buttons(page_state){
    html_str = "";
    var location = page_state.getButtonsContainer();
    var dicts = page_state.getButtons();
    var opts = page_state.getOptions();

    for(var i = 0; i < dicts.length; i++){
        var dict = dicts[i];
        html_str += '<p><div class="row"><div class="col-2"><span class="label label-primary"><strong>'+opts[i]+'</strong></span></div><div class="col-10"><div class="btn-group btn-group-sm" role="group">';

        for(var key in dict){
            var val = dict[key];
            if(key == page_state.getPageContentItem(i+1)){
                html_str += '<button type="button" class="btn btn-primary active" onclick="reload_page_from_element(this,page_state)" data-index="'+(i+1)+'" data-opt="'+opts[i]+'" data-val="'+key+'">'+val+'</button>';
            }
            else{
                html_str += '<button type="button" class="btn btn-primary" onclick="reload_page_from_element(this,page_state)" data-index="'+(i+1)+'" data-opt="'+opts[i]+'" data-val="'+key+'">'+val+'</button>';
            }
        }
        html_str += "</div></div></div></p>"
    }

    location.html(html_str)
}

function load_results(page_state){
    var results_content = page_state.getPageContent();
    var location = page_state.getPageContainer();
    var loctext = page_state.getTextContainer();
    var result_str = "results/";
    var buttons = page_state.getButtons();
    var dim_chart_flag = results_content[0] == "dim" && results_content[4] == 'chart'; //Special case: dim chart
    
    for(var i = 0; i < results_content.length; i++){
        if(!page_state.isSpecialItem(i)) result_str += results_content[i]
        if(i < results_content.length-1 && !page_state.isSpecialItem(i+1)) result_str += "-";
    }
    result_str += ".html";

    //loctext.empty();
    loctext.load("view/"+results_content[0]+".html");

    location.load(result_str, function(){
        var tables_max = document.getElementsByClassName("maxhighlightable");
        highlight_max(tables_max);

        var tables_avg = document.getElementsByClassName("withavg");
        insert_avg_into_tfoot(tables_avg);
    });

    if(dim_chart_flag){
        location.load(result_str, function(){
            var table = document.getElementsByClassName("dimchartable").item(0);
            var data = table_to_highchart_data(table,results_content[2],page_state);
            
            Highcharts.chart('page-content', {
                chart: {
                    zoomType: "xy"
                },
                title: {
                    text: buttons[0][results_content[1]] + " - " + buttons[1][results_content[2]] + " - " + buttons[2][results_content[3]]
                },

                subtitle: {
                    text: 'Select an area to zoom in.'
                },

                yAxis: {
                    title: {
                        text: 'Score'
                    }
                },
                xAxis: {
                    allowDecimals: false,
                    title: {
                        text: 'Dimension'
                    }
                },
                legend: {
                    layout: 'vertical',
                    align: 'right',
                    verticalAlign: 'middle'
                },

                plotOptions: {
                    series: {
                        label: {
                            connectorAllowed: true
                        },
                        pointStart: 1
                    }
                },

                series: data,

                responsive: {
                    rules: [{
                        condition: {
                            maxWidth: 500
                        },
                        chartOptions: {
                            legend: {
                                layout: 'horizontal',
                                align: 'center',
                                verticalAlign: 'bottom'
                            }
                        }
                    }]
                }

            });
        });
    }
}

function reload_page(page_state){
    var topic = page_state.getPageContentItem(0);
    if(topic == "home"){
        page_state.getTextContainer().load("view/home.html");
    }
    else{
        load_results(page_state);
    }
}

function reload_page_from_element(elmnt, page_state){
    var index = elmnt.getAttribute("data-index");
    var val = elmnt.getAttribute("data-val");
    page_state.setPageContentItem(index,val);
    load_buttons(page_state);
    reload_page(page_state);
}

function reload_page_from_menu(elmnt, page_state){
    var id = elmnt.getAttribute("id");
    page_state.setPageContent([id]);
    load_buttons(page_state);
    reload_page(page_state);
}
