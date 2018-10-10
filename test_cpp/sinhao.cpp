s a_finished = 0;
s b_finished = 0;
s c_finished = 0;
s d_finished = 0;
s e_finished = 0;



void A(){
    
    /* a */

    V(a_finished);
    V(a_finished);
}

void B(){
    P(a_finished);

    /* b  */

    V(b_finished);

}

void C(){
    P(a_finished);

    /* c */

    V(c_finished);
}

void D(){
    P(b_finished);

    /* d */

    V(d_finished);
}

void E(){
    P(d_finished);
    P(c_finished);

    /* e */
}

void main(){
    parbegin(A(),B(),C(),D(),E());
}