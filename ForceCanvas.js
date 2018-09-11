//Programmer: Chris Tralie

/** A canvas for drawing a self-similarity matrix synchronized with audio */

/**
 * Create a structure canvas
 * @param {object} audio_obj - A dictionary of audio parameters, including 
 * 	a handle to the audio widget and a time interval between adjacent rows
 *  of the SSM, and the dimension "dim" of the SSM
 */
function ForceCanvas(audio_obj) {
	this.graphcanvas = d3.select("#ForceCanvas");
	this.audio_obj = audio_obj;
	this.fac = 1; //Downsampling factor from full SSM image
	this.nodes = [];
	/**
	 * A function use to update the images on the canvas and to
	 * resize things accordingly
	 * @param {object} params : A dictionary of parameters returned
	 * from the Python program as a JSON file
	 */

	this.dragstarted = function(d) {
		if (!d3.event.active) this.simulation.alphaTarget(0.3).restart();
		d.fx = d.x;
		d.fy = d.y;
	};

	this.dragged = function(d) {
		d.fx = d3.event.x;
		d.fy = d3.event.y;
	};

	this.dragended = function(d) {
		if (!d3.event.active) this.simulation.alphaTarget(0);
		d.fx = null;
		d.fy = null;
	};

	this.updateParams = function(params) {
		// With heavy inspiration from https://bl.ocks.org/mbostock/4062045
		var width = params.dim;
		var height = params.dim;

		this.simulation = d3.forceSimulation()
							.force("link", d3.forceLink().id(function(d) { return d.id; }))
							.force("charge", d3.forceManyBody())
							.force("center", d3.forceCenter(width / 2, height / 2));
		
		graph = JSON.parse(params.graph);
		this.fac = graph.fac;

		// Clear all graph elements if any exist
		this.graphcanvas.selectAll("*").remove();
		this.graphcanvas.attr('width', params.dim).attr('height', params.dim);
		
		var link = this.graphcanvas.append("g")
			.attr("class", "links")
			.selectAll("line")
			.data(graph.links)
			.enter().append("line")
			.attr("stroke-width", function(d) { return Math.sqrt(d.value); });
		
		this.nodes = this.graphcanvas.append("g")
			.attr("class", "nodes")
			.selectAll("circle")
			.data(graph.nodes)
			.enter().append("circle")
			.attr("r", 5)
			.attr("fill", function(d, i) { 
				var c = d.color;
				return d3.rgb(c[0], c[1], c[2]); 
			})
			.call(d3.drag()
				.on("start", this.dragstarted.bind(this))
				.on("drag", this.dragged.bind(this))
				.on("end", this.dragended.bind(this)));
		
		this.simulation
			.nodes(graph.nodes)
			.on("tick", ticked.bind(this));
		
		this.simulation.force("link")
			.links(graph.links);
		
		function ticked() {
			link
				.attr("x1", function(d) { return d.source.x; })
				.attr("y1", function(d) { return d.source.y; })
				.attr("x2", function(d) { return d.target.x; })
				.attr("y2", function(d) { return d.target.y; });
		
			this.nodes
				.attr("cx", function(d) { return d.x; })
				.attr("cy", function(d) { return d.y; });
		}
	};

	/**
	 * A function which toggles all of the visible elements to show
	 */
	this.show = function() {
		this.graphcanvas.style("display", "block");
	};

	/**
	 * A function which toggles all of the visible elements to hide
	 */
	this.hide = function() {
		this.graphcanvas.style("display", "none");
	};


	/**
	 * A fuction which highlights the node in the graph with the time closest
	 * to the current play time in the audio
	 */
	this.updateCanvas = function() {
		if (this.audio_obj.time_interval > 0) {
			var idx = this.audio_obj.audio_widget.currentTime / this.audio_obj.time_interval;
			idx = Math.round(idx/this.fac);
			console.log(idx);
			this.nodes.attr("r", 
				function(d, i) {
					if (i == idx) {
						return 20;
					}
					return 5;
				});
		}
	};
	
	/**
	 * A function that should be called in conjunction with requestionAnimationFrame
	 * to refresh this canvas.  It continually highlights the node corresponding to
	 * the the closest position in audio.  Continually generates callbacks as long as the
	 * audio is playing, but stops generating callbacks when it is paused to save
	 * computation
	 */
	this.repaint = function() {
		this.updateCanvas();
		if (!this.audio_obj.audio_widget.paused) {
			requestAnimationFrame(this.repaint.bind(this));
		}
	};

}