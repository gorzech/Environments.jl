function double_inverted(F, a1, a2, a1_t, a2_t, g, l, m_cart, m_pole)
    sa1, ca1 = sincos(a1)
    sa2, ca2 = sincos(a2)
    sa12 = ca2 * sa1 - ca1 * sa2
    ca12 = sa1 * sa2 + ca1 * ca2

    A = SA[
        m_cart+m_pole*2.0 l*m_pole*ca1*(3.0/2.0) (l*m_pole*ca2)/2.0
        l*m_pole*ca1*(3.0/2.0) l^2*m_pole*(4.0/3.0) (l^2*m_pole*ca12)/2.0
        (l*m_pole*ca2)/2.0 (l^2*m_pole*ca12)/2.0 (l^2*m_pole)/3.0
    ]
    b = SA[
        F+a1_t^2*l*m_pole*sa1*(3.0/2.0)+(a2_t^2*l*m_pole*sa2)/2.0,
        a2_t^2*l^2*m_pole*sa12*(-1.0/2.0)+g*l*m_pole*sa1*(3.0/2.0),
        (a1_t^2*l^2*m_pole*sa12)/2.0+(g*l*m_pole*sa2)/2.0,
    ]
    x = A \ b
    x[1], x[2], x[3]
end
