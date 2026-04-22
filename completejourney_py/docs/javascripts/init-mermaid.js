// Additional mermaid configuration for mkdocs-jupyter compatibility
window.addEventListener('load', function() {
    // Wait a bit for mkdocs-jupyter to process first
    setTimeout(function() {
        if (typeof mermaid !== 'undefined') {
            // Re-initialize with our preferred settings
            mermaid.initialize({
                startOnLoad: false,
                theme: 'default',
                maxTextSize: 100000,
                maxEdges: 100000,
                fontFamily: 'monospace',
                themeVariables: {
                    primaryColor: '#2196F3',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#1976D2',
                    lineColor: '#757575'
                }
            });
            
            // Find any unprocessed mermaid diagrams
            const unprocessedMermaid = document.querySelectorAll('.jp-Mermaid pre.mermaid:not([data-processed])');
            if (unprocessedMermaid.length > 0) {
                console.log('Found unprocessed mermaid diagrams:', unprocessedMermaid.length);
                
                unprocessedMermaid.forEach(function(element, index) {
                    try {
                        element.setAttribute('data-processed', 'true');
                        mermaid.init(undefined, element);
                    } catch (error) {
                        console.error('Mermaid rendering error:', error);
                    }
                });
            }
        }
    }, 1000);
});