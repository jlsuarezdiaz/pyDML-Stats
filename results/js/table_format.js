function highlight_max(tables){
    for(var t = 0; t < tables.length; t++){
        var table = tables.item(t);

        //var table = document.getElementById("the-table");
        for(var i = 1, row; row = table.rows[i]; i++){
            var sign = 1;
            var head_col = row.cells[0];
            if(head_col.innerHTML == "AVG RANKING"){
                sign = -1;
            }
            var max = sign*-1e20;
            //var jmax = -1;
            for(var j = 1, col; col = row.cells[j]; j++){
                var val = col.innerHTML;
                if(val == "NaN"){
                    col.innerHTML = "-";
                    col.style.color = "red";
                }
                else{
                    val = Number(val);
                    if(!isNaN(val) && (sign == 1 && val > max) || (sign == -1 && val < max)){
                        max = val;
                        //jmax = j;
                    }
                }
            }

            for(var j = 0, col; col = row.cells[j]; j++){ // Para múltiples máximos
                var val = col.innerHTML;
                if(val == max){
                    col.style.fontWeight = "bold";
                }
            }
            //row.cells[jmax].style.fontWeight = "bold";
        }

    }
}

function insert_avg_into_tfoot(tables){
    for(var t = 0; t < tables.length; t++){
        var table = tables.item(t);
        var foot = table.createTFoot();
        foot.style.backgroundColor = "#ffffe6";
        //foot.className += " thead-dark";
        for(var i = table.rows.length-1, row; (row = table.rows[i]) && (i > 0); i--){
            var head_col = row.cells[0].innerHTML;
            if(head_col.startsWith("AVG")){
                foot.insertBefore(row,foot.firstChild);
                //var newNode = document.createElement("tfoot");
                //var parent = row.parentNode;
                //parent.insertBefore(newNode,row);
            }
        }

    }
}



function table_to_highchart_data(table,dataset,page_state){
    var data=[];
    var nrow = table.rows.length;
    var ncol= table.rows[0].cells.length;
    for(var i = 1; i < ncol; i++){
        var str = table.rows[0].cells[i].innerHTML;
        var coldata = [];
        for(var j = 1; j < nrow; j++){
            var txtdim = table.rows[j].cells[0].innerHTML;
            var dim;
            if(txtdim == "N. Classes - 1"){
                dim = page_state.getMetadata(dataset+"-nclasses")-1;
            }
            else if(txtdim == "Max. Dimension"){
                dim = page_state.getMetadata(dataset+"-maxdim");
            }
            else{
                dim = Number(txtdim);
            }
            var val = Number(table.rows[j].cells[i].innerHTML);
            coldata.push([dim,val]);
        }

        coldata.sort(function(a,b){
            if (a[0] === b[0]) {
                return 0;
            }
            else {
                return (a[0] < b[0]) ? -1 : 1;
            }
        });
        var ret = [coldata[0]];
        for (var j = 1; j < coldata.length; j++) { 
            if (coldata[j-1][0] != coldata[j][0] || coldata[j-1][1] != coldata[j][1]) {
                ret.push(coldata[j]);
            }
        }
        coldata = ret;
        
        data.push({name:str, data:coldata});
    }
    return data;
}