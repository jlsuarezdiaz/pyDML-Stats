class PageState{
    constructor(){
        this.page_content = [];
        this.buttons = [];
        this.page_content[0] = "home";
        this.opts = [];
        this.metadata={};
        this.special = [];
    }

    getPageContent(){
        return this.page_content;
    }

    getPageContentItem(i){
        return this.page_content[i];
    }

    setPageContentItem(i,val){
        this.page_content[i] = val;
    }

    setPageContent(content){
        this.page_content = content;
        this.completePageContent();
    }

    completePageContent(){
        var topic = this.page_content[0];
        if(topic == "home"){
            this.buttons = [];
            this.opts = [];
            this.special = [false];
        }
        else if(topic == "datasets"){
            this.buttons = [];
            this.opts = [];
            this.special = [false];
        }
        else if(topic == "basic"){
            this.page_content[1] = "3nn";
            this.page_content[2] = "train";
            this.buttons = [{'3nn':'3-NN','5nn':'5-NN','7nn':'7-NN'},{'train':'Train','test':'Test'}];
            this.opts = ['Classifier', 'Subset'];
            this.special = [false,false,false];
        }
        else if(topic == "ncm"){
            this.page_content[1] = "train";
            this.buttons = [{'train':'Train','test':'Test'}];
            this.opts = ['Subset'];
            this.special = [false,false];
        }
        else if(topic == "ker"){
            this.page_content[1] = "3nn";
            this.page_content[2] = "train";
            this.buttons = [{'3nn':'3-NN'},{'train':'Train','test':'Test'}];
            this.opts = ['Classifier', 'Subset'];
            this.special = [false,false,false];
        }
        else if(topic == "dim"){
            this.page_content[1] = "3nn";
            this.page_content[2] = "sonar";
            this.page_content[3] = "train";
            this.page_content[4] = "table";
            this.special = [false,false,false,false,true]
            this.buttons = [{'3nn':'3-NN','5nn':'5-NN','7nn':'7-NN'},{'sonar':'Sonar','movement_libras':'Movement Libras','spambase':'Spambase'},{'train':'Train','test':'Test'},{'table':'Table','chart':'Line chart'}];
            this.opts = ['Classifier', 'Dataset', 'Subset','View'];
        }
    }

    setPageContainer(obj){
        this.page_container = obj;
    }

    getPageContainer(){
        return this.page_container;
    }

    setButtonsContainer(obj){
        this.buttons_container = obj;
    }

    getButtonsContainer(){
        return this.buttons_container;
    }

    setTextContainer(obj){
        this.text_container = obj;
    }

    getTextContainer(){
        return this.text_container;
    }

    getButtons(){
        return this.buttons;
    }

    getOptions(){
        return this.opts;
    }

    addMetadata(key,val){
        this.metadata[key] = val;
    }

    getMetadata(key){
        return this.metadata[key];
    }

    isSpecialItem(item){
        return this.special[item];
    }

}